import os
from typing import Any

import cv2
import habitat
import numpy as np
from habitat import RLEnv
from habitat.utils.visualizations import maps
from habitat_baselines.common.baseline_registry import baseline_registry
from scipy.spatial.distance import cosine

from experimenting_env.sensor_data import AgentPoseSense
from experimenting_env.utils import projection_utils as pu
from experimenting_env.utils.matching import get_objects_ids

from .env_base import BaseEnv
from .sensors import *


@baseline_registry.register_env(name="sealenv-v0")
class SealENV(BaseEnv):
    def __init__(self, config, **kwargs):
        super().__init__(
            config,
            **kwargs,
        )
        self.explored_area = 0
        self.class_features = {}
        self.model = None
        self._max_episode_steps = self._env._max_episode_steps
        self.pcd = None
        self.last_reward = 0.0
        self.counter = 0

    def is_done(self):
        print("calling is done:", self.episode_over)
        return self.episode_over

    def set_model(self, model):
        self.model = model

    def _past_limit(self) -> bool:
        if (
            self._max_episode_steps != 0
            and self._max_episode_steps <= self._elapsed_steps
        ):
            return True

        return False

    def get_episode_step(self):
        return self._elapsed_steps
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

        if unknown_percent < 0.2:
            print("Exploration task completed")
            self.episode_over = True

        self.episode_over += self._past_limit()
        return self.episode_over

    def _episode_success(self, observations):
        r"""Returns True if within distance threshold of the goal."""
        return False

    def get_info(self, observations):
        info = self.habitat_env.get_metrics()
        info["episode_success"] = self._episode_success(observations)
        return info

    def step(self, action):
        self._elapsed_steps += 1
        self._previous_action = action
        observations = [
            self.habitat_env.step(action)
        ]  # single cam, but still expect to receive a list [ob1, ... obn] where i = idx of camera

        lower_bound, upper_bound = self._env._sim.pathfinder.get_bounds()
        #self._update_pointcloud(observations)

        done = self.get_done(observations)
        info = self.get_info(observations)
        reward = self.get_reward()
        self.last_reward = reward
        return observations, reward, done, info

    def reset(self):
        self.episode_over = False
        obs = super().reset()
        self.explored_area = 0
        self.last_reward = 0.0
        self._elapsed_steps = 0

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

        self.pcd.update_logits(preds, infos)
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
        

    def get_pcd(self):
        pcd = torch.tensor(
            self.pcd.current_connected_pcd / self.pcd.current_connected_pcd.max()
        ).unsqueeze(0)
        return pcd
    def get_reward(self):
        reward = 0
        if self.pcd is not None:
            for k, l in self.pcd.object_id_to_logits.items():
                score = l.max()
                if score > 0.9:
                    reward += (self.pcd.infos[:, 2] == k).sum()

        self.last_reward = reward

        
        print(f"Reward {reward}")
        return reward

    def get_reward_range(self):
        """This function is not used, Habitat-RLEnv requires this function"""

        return (-1.0, 1.0)
