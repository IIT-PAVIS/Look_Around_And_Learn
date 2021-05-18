import os
import time
from typing import Any

import cv2
import habitat
import numpy as np
from habitat import RLEnv
from habitat.utils.visualizations import maps
from habitat_baselines.common.baseline_registry import baseline_registry
from scipy.spatial.distance import cosine
from torch.nn import functional as F

from experimenting_env.sensor_data import AgentPoseSense
from experimenting_env.utils import projection_utils as pu
from experimenting_env.utils.matching import get_objects_ids
from experimenting_env.utils.model import get_grid

from ..agents.neural_slam.utils import HabitatMaps
from .env_base import BaseEnv
from .sensors import *


@baseline_registry.register_env(name="SemanticDisagreement-v0")
class SemanticDisagreement(BaseEnv):
    def __init__(self, config, **kwargs):
        super().__init__(
            config,
            **kwargs,
        )
        self.class_features = {}
        self.model = None
        self.pcd = None
        self.disagreement_map = None
        self.last_reward = 0.0
        self._episode_sum_reward = 0.0

    def is_done(self):

        return self.episode_over

    def set_model(self, model):
        self.model = model

    def get_done(self, observations):
        self.episode_over = False

        info = self.habitat_env.get_metrics()
        allo_map = info['top_down_map']['map']
        allo_map[allo_map > 1] = 1
        expl_map = info['top_down_map']['fog_of_war_mask']

        unknown_map = allo_map - expl_map
        freespace_area = np.sum(allo_map[allo_map == 1])
        unknown_area = np.sum(unknown_map[unknown_map > 0])
        unknown_percent = unknown_area / freespace_area

        # if unknown_percent < 0.1:
        #     print("Exploration task completed")
        #     self.episode_over = True

        self.episode_over = self._past_limit()
        return self.episode_over

    def _episode_success(self, observations):
        r"""Returns True if within distance threshold of the goal."""
        return False

    def get_info(self, observations):
        info = self.habitat_env.get_metrics()
        info["episode_success"] = self._episode_success(observations)
        info["episode_reward"] = self._episode_sum_reward / self._max_episode_steps

        expl_map = info['top_down_map']['fog_of_war_mask']
        allo_map = info['top_down_map']['map']
        allo_map[allo_map > 1] = 1
        info["area_ratio"] = expl_map.sum() / (allo_map > 0).sum()

        self.area_ratio = expl_map.sum() / (allo_map > 0).sum()

        return info

    def get_episode_step(self):
        return self._elapsed_steps

    def step(self, action):
        observations = [super().step(action)]

        done = self.get_done(observations)
        info = self.get_info(observations)
        reward = (
            self.get_reward()
        )  # get reward uses current pcd for getting reward; if the pcd is update (e.g., with _update_pointcloud, after getting current observations) you need to call get_reward again

        return observations, reward, done, info

    def reset(self):
        obs = super().reset()
        self.last_reward = 0.0
        self._elapsed_steps = 0
        self._episode_sum_reward = 0.0

        return [obs]

    def _update_pointcloud(self, observations):

        instances = observations[0]['bbs']
        preds = instances['instances'].to("cpu")
        batch = observations[0]['bbsgt']
        batch['episode'] = int(self._env.current_episode.episode_id)

        infos = get_objects_ids([batch], [preds])[0]

        preds.infos = infos

        pose = observations[0]['position']

        point_cloud = pu.project_semantic_masks_to_3d(
            observations[0]['depth'].squeeze(),
            pose,
            preds,
            infos,
        )
        point_cloud._episode = int(self._env.current_episode.episode_id)

        lower_bound, upper_bound = self.get_map_bounds()
        if self.pcd is None or self.pcd._episode != point_cloud._episode:
            self.pcd: pu.SemanticPointCloud = point_cloud

        else:
            point_cloud.preprocess(lower_bound, upper_bound)

            self.pcd += point_cloud

        if len(preds) == 0:
            return

        t = time.time()
        self.pcd.preprocess(lower_bound, upper_bound)

    def to_grid(self, realworld_x, realworld_y, grid_resolution):

        lower_bound, upper_bound = self.get_map_bounds()

        grid_size = (
            abs(upper_bound[2] - lower_bound[2]) / grid_resolution[0],
            abs(upper_bound[0] - lower_bound[0]) / grid_resolution[1],
        )
        grid_x = int((lower_bound[2] + realworld_x) / grid_size[0])
        grid_y = int((lower_bound[0] + realworld_y) / grid_size[1])
        return grid_x, grid_y

    def get_and_update_disagreement_map(self):

        lower_bound, upper_bound = self._env._sim.pathfinder.get_bounds()

        disagreement_map = self.pcd.get_topdown_semantic(lower_bound, upper_bound)[
            :, :, -1
        ]
        self.disagreement_map = disagreement_map
        return disagreement_map

    
    def get_reward(self):

        disagreement_reward = 0

        if self.disagreement_map is not None:
            disagreement_reward = self.disagreement_map.sum() / 1000.0

        self.last_reward = disagreement_reward
        self._episode_sum_reward += disagreement_reward

        return disagreement_reward

    def get_reward_range(self):
        """This function is not used, Habitat-RLEnv requires this function"""

        return (-1.0, 1.0)

    def get_area_ratio(self):
        return self.area_ratio

    def _get_gt_map(self, full_map_size):

        # Get map in habitat simulator coordinates
        self.map_obj = HabitatMaps(self.habitat_env)
        if self.map_obj.size[0] < 1 or self.map_obj.size[1] < 1:
            log.error("Invalid map")
            return None

        agent_y = self._env.sim.get_agent_state().position.tolist()[1] * 100.0
        sim_map = self.map_obj.get_map(agent_y, -50.0, 50.0)

        sim_map[sim_map > 0] = 1.0

        # Transform the map to align with the agent
        min_x, min_y = self.map_obj.origin / 100.0
        x, y, o = self.get_sim_location()
        x, y = -x - min_x, -y - min_y
        range_x, range_y = self.map_obj.max / 100.0 - self.map_obj.origin / 100.0

        map_size = sim_map.shape
        scale = 2.0
        grid_size = int(scale * max(map_size))
        grid_map = np.zeros((grid_size, grid_size))

        grid_map[
            (grid_size - map_size[0]) // 2 : (grid_size - map_size[0]) // 2
            + map_size[0],
            (grid_size - map_size[1]) // 2 : (grid_size - map_size[1]) // 2
            + map_size[1],
        ] = sim_map

        if map_size[0] > map_size[1]:
            st = torch.tensor(
                [
                    [
                        (x - range_x / 2.0)
                        * 2.0
                        / (range_x * scale)
                        * map_size[1]
                        * 1.0
                        / map_size[0],
                        (y - range_y / 2.0) * 2.0 / (range_y * scale),
                        180.0 + np.rad2deg(o),
                    ]
                ]
            )

        else:
            st = torch.tensor(
                [
                    [
                        (x - range_x / 2.0) * 2.0 / (range_x * scale),
                        (y - range_y / 2.0)
                        * 2.0
                        / (range_y * scale)
                        * map_size[0]
                        * 1.0
                        / map_size[1],
                        180.0 + np.rad2deg(o),
                    ]
                ]
            )

        rot_mat, trans_mat = get_grid(
            st, (1, 1, grid_size, grid_size), torch.device("cpu")
        )

        grid_map = torch.from_numpy(grid_map).float()
        grid_map = grid_map.unsqueeze(0).unsqueeze(0)
        translated = F.grid_sample(grid_map, trans_mat)
        rotated = F.grid_sample(translated, rot_mat)

        episode_map = torch.zeros((full_map_size, full_map_size)).float()
        if full_map_size > grid_size:
            episode_map[
                (full_map_size - grid_size) // 2 : (full_map_size - grid_size) // 2
                + grid_size,
                (full_map_size - grid_size) // 2 : (full_map_size - grid_size) // 2
                + grid_size,
            ] = rotated[0, 0]
        else:
            episode_map = rotated[
                0,
                0,
                (grid_size - full_map_size) // 2 : (grid_size - full_map_size) // 2
                + full_map_size,
                (grid_size - full_map_size) // 2 : (grid_size - full_map_size) // 2
                + full_map_size,
            ]

        episode_map = episode_map.numpy()
        episode_map[episode_map > 0] = 1.0

        return episode_map

    def get_sim_location(self):
        agent_state = super().habitat_env.sim.get_agent_state(0)
        x = -agent_state.position[2]
        y = -agent_state.position[0]
        axis = quaternion.as_euler_angles(agent_state.rotation)[0]
        if (axis % (2 * np.pi)) < 0.1 or (axis % (2 * np.pi)) > 2 * np.pi - 0.1:
            o = quaternion.as_euler_angles(agent_state.rotation)[1]
        else:
            o = 2 * np.pi - quaternion.as_euler_angles(agent_state.rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        return x, y, o


    def get_kl_predictions(self, observations):
        instances = observations[0]['bbs']
        preds = instances['instances'].to("cpu")
        batch = observations[0]['bbsgt']
        batch['episode'] = int(self._env.current_episode.episode_id)
        pose = observations[0]['position']
        infos = get_objects_ids([batch], [preds])[0]
        preds.infos = infos
        depth = observations[0]['depth'].squeeze()
        
        kl = self.pcd._get_kl(
            depth,
            pose,
            preds,
            infos)

        return kl.sum()
