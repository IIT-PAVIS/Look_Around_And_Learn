# type: ignore
import copy
import gc
import os
from collections import deque
from typing import Any, Dict, Union

import cv2
import habitat
import habitat_sim
import magnum as mn
import numpy as np
from habitat.core.simulator import Observations
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.nav import NavigationGoal, NavigationTask
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat_baselines.agents.simple_agents import GoalFollower, RandomAgent
from habitat_baselines.common.base_trainer import BaseRLTrainer
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_sim.utils import common as utils

from experimenting_env.agents.ppo_trainer import PPOTrainer
from experimenting_env.utils.astar2 import Grid, astar
from experimenting_env.utils.deprecated import *
from experimenting_env.utils.habitat_utils import (
    construct_envs,
    get_unique_scene_envs_generator,
)
from experimenting_env.utils.sensors_utils import save_obs
from experimenting_env.utils.skeleton import *


class Baseline(BaseRLTrainer):
    def __init__(self, config, exp_base_dir, agent_class, **kwargs):
        super().__init__(config)

        self.config = config
        self.kwargs = kwargs

        self.agent = agent_class(
            self.config.TASK.SUCCESS.SUCCESS_DISTANCE, self.config.TASK.GOAL_SENSOR_UUID
        )

        self.exp_path = os.path.join(os.getcwd(), "dataset")

    def _init_train(self):
        envs = self._init_envs()
        os.makedirs(self.exp_path, exist_ok=True)

        self.current_observations = envs.reset()
        self.last_observations = []
        self.current_dones = []
        self.current_infos = []
        self.current_rewards = []
        return envs

    def _step(self, envs):
        actions = []
        for index_env in range(envs.num_envs):
            act = self.agent.act(self.current_observations[index_env][0])
            actions.append(act)
            envs.async_step_at(index_env, act)

        results = [envs.wait_step_at(index_env) for index_env in range(envs.num_envs)]

        self.last_observations = self.current_observations
        self.current_observations = [r[0] for r in results]
        self.last_actions = actions
        self.current_rewards = [r[1] for r in results]
        self.current_dones = [r[2] for r in results]
        self.current_infos = [r[3] for r in results]

        self.current_steps += 1

    def _init_envs(self, config=None, kwargs=None):
        if config is None:
            config = self.config
        if kwargs is None:
            kwargs = self.kwargs
        self.num_steps_done = 0
        envs = construct_envs(config, get_env_class(config.ENV_NAME), True, **kwargs)
        self.current_steps = np.zeros(envs.num_envs)
        return envs

    def train(self) -> None:
        pass

    def generate(self, config=None, kwargs=None) -> None:
        envs = self._init_train()

        generated_observations_paths = []
        self.exp_path = os.path.join(self.exp_path)
        while not self.is_done():

            self._step(envs)
            for idx in range(envs.num_envs):
                obs = self.current_observations[idx]
                
                done = self.current_dones[idx]
                episode = envs.current_episodes()[idx]

                paths = save_obs(
                    self.exp_path, episode.episode_id, obs, self.current_steps[idx]
                )
                
                generated_observations_paths.append(paths)
                self.num_steps_done += 1
                if done:
                    self.current_steps[idx] = 0
            if self.num_steps_done % 10 == 0:
                print(f"Exploration at {int(self.percent_done() * 100)}%")
        del self.current_dones
        del self.current_observations

        envs.close()
        return sorted(generated_observations_paths)


@baseline_registry.register_trainer(name="randombaseline")
class RandomBaseline(Baseline):
    def __init__(self, config, exp_base_dir, **kwargs):
        super().__init__(config, exp_base_dir, RandomAgent, **kwargs)


class BounceAgent(RandomAgent):
    def __init__(self, success_distance: float, goal_sensor_uuid: str) -> None:
        super().__init__(success_distance, goal_sensor_uuid)
        self.turn_count = 0

    def act(self, observations: Observations) -> Dict[str, Union[int, int]]:
        action = HabitatSimActions.MOVE_FORWARD

        # if collision with navmesh and not turning already
        if observations['agent_collision_sensor'] and self.turn_count == 0:
            self.turn_count = 16  # depends on TURN_ANGLE (where is it declared?)
            print("Collision:", observations['agent_collision_sensor'])

        if self.turn_count > 1:
            action = (
                HabitatSimActions.TURN_LEFT
            )  # TODO: choose turning side based on tangent angle wrt the obstacle

            self.turn_count -= 1
        elif self.turn_count == 1:
            action = HabitatSimActions.MOVE_FORWARD
            self.turn_count -= 1

        return {"action": action}


@baseline_registry.register_trainer(name="bouncebaseline")
class BounceBaseline(Baseline):
    def __init__(self, config, exp_base_dir, **kwargs):
        super().__init__(config, exp_base_dir, BounceAgent, **kwargs)


class RotateAgent(RandomAgent):
    def __init__(self, success_distance: float, goal_sensor_uuid: str) -> None:
        super().__init__(success_distance, goal_sensor_uuid)

    def act(self, observations: Observations) -> Dict[str, Union[int, int]]:
        return {"action": HabitatSimActions.TURN_LEFT}


@baseline_registry.register_trainer(name="rotatebaseline")
class RotateBaseline(Baseline):
    def __init__(self, config, exp_base_dir, **kwargs):
        super().__init__(config, exp_base_dir, RotateAgent, **kwargs)


@baseline_registry.register_task(name="FrontExp-v0")
class FrontierExplorationTask(NavigationTask):
    def __init__(self, config, sim, dataset):
        super().__init__(config=config, sim=sim, dataset=dataset)
        self.sim = sim

    def _check_episode_is_active(self, *args: Any, **kwargs: Any) -> bool:
        return True


@baseline_registry.register_trainer(name="randomgoalsbaseline")
class RandomGoalsBaseline(Baseline):
    def __init__(self, config, exp_base_dir, **kwargs):
        super().__init__(config, exp_base_dir, GoalFollower, **kwargs)
        self.config = config

        self.visualize = False  # self.config.randomgoals.visualize  # visualize maps

    def goto_next_subgoal(self, idx):
        if (self.got_new_plan[idx] or self.current_steps[idx] % 10 == 0) and len(
            self.sub_goals[idx]
        ) > 0:
            new_sub_goal = self.sub_goals[idx].pop(-1)
            self.got_new_plan[idx] = False
            print("Picking next subgoal. Remaining subgoals=", len(self.sub_goals[idx]))
            self.envs.call_at(
                idx, "set_goals", {"data": [NavigationGoal(position=new_sub_goal)]}
            )

    def compute_new_goals(self):
        if (
            self.first_step
            or self.current_steps[0] % self.config.randomgoals.replanning_steps == 0
        ):
            print("Computing new goals...")
            self.first_step = False

            cpu_actions = np.random.uniform(size=(self.envs.num_envs, 2))

            # get the current map
            mymaps = [
                self.current_infos[idx]["top_down_map"]['map'].copy() * 255
                for idx in range(self.envs.num_envs)
            ]

            mymaps_sizes_pixels = [x.shape for x in mymaps]

            for action in cpu_actions:
                print("goal:", action)

            rescaled_pixel_goals = [
                [
                    int(
                        action[0]
                        * mymaps_sizes_pixels[i][1]
                        # * self.current_observations[i][0]['disagreement_map'].shape[1]
                    ),
                    int(
                        action[1]
                        * mymaps_sizes_pixels[i][0]
                        # * self.current_observations[i][0]['disagreement_map'].shape[0]
                    ),
                ]
                for i, action in enumerate(cpu_actions)
            ]
            # compute subgoals for each goal
            for idx, pixel_goal in enumerate(rescaled_pixel_goals):
                # convert pixels to global map coords
                lower_bound, upper_bound = self.envs.call_at(idx, "get_map_bounds")
                grid_resolution = (
                    abs(upper_bound[2] - lower_bound[2]) / mymaps_sizes_pixels[idx][0],
                    abs(upper_bound[0] - lower_bound[0]) / mymaps_sizes_pixels[idx][1],
                )

                # lower_bounds are inverted, why?!
                realworld_x = lower_bound[0] + pixel_goal[0] * grid_resolution[0]
                realworld_y = lower_bound[2] + pixel_goal[1] * grid_resolution[1]

                goal_world = [
                    realworld_x,
                    realworld_y,
                ]  # goal in world coords (m)

                agent_pos = [
                    self.current_observations[idx][0]['position']['position'][2],
                    self.current_observations[idx][0]['position']['position'][0],
                ]

                scaled_down_map = mymaps[idx]  # no rescalement required

                grid_size = (
                    abs(upper_bound[2] - lower_bound[2]) / scaled_down_map.shape[0],
                    abs(upper_bound[0] - lower_bound[0]) / scaled_down_map.shape[1],
                )

                grid_x = int((agent_pos[1] - lower_bound[0]) / grid_size[0])
                grid_y = int((agent_pos[0] - lower_bound[2]) / grid_size[1])

                # set start end end pose for A*
                start_pos = [grid_x, grid_y]
                end_pos = pixel_goal

                end_pos[0] = max(0, min(end_pos[0], scaled_down_map.shape[1] - 1))
                end_pos[1] = max(0, min(end_pos[1], scaled_down_map.shape[0] - 1))

                # extract a path to the goal (path is in opencv pixel coords)
                path, img = do_plan(
                    scaled_down_map, start_pos, end_pos, False, True, end_pos
                )

                if len(path) == 0:
                    print("No path!")
                    self.sub_goals[idx] = []
                    continue
                path.insert(0, end_pos)

                if self.visualize:
                    self.astar_img[idx] = img.copy()
                    cv2.circle(
                        self.astar_img[idx],
                        (end_pos[0], end_pos[1]),
                        20,
                        (255, 255, 0),
                        4,
                    )

                # extract sub-goals
                self.sub_goals[idx] = []
                for subgoal in path:
                    realworld_subgoal_x = lower_bound[0] + subgoal[0] * (
                        abs(upper_bound[0] - lower_bound[0]) / scaled_down_map.shape[1]
                    )
                    realworld_subgoal_y = lower_bound[2] + subgoal[1] * (
                        abs(upper_bound[2] - lower_bound[2]) / scaled_down_map.shape[0]
                    )
                    realworld_subgoal = [
                        realworld_subgoal_x,
                        self.current_observations[idx][0]['position']['position'][1],
                        realworld_subgoal_y,
                    ]

                    self.sub_goals[idx].append(realworld_subgoal)

                self.sub_goals[idx].pop(-1)

                if len(self.sub_goals[idx]) > 0:
                    self.got_new_plan[idx] = True
                else:
                    print("A* failed, no waypoints")

    def generate(self) -> None:
        self.envs = self._init_train()
        generated_observations_paths = []
        self.exp_path = os.path.join(self.exp_path)

        self.sub_goals = [[] for _ in range(self.envs.num_envs)]
        self.sub_goals_counter = [0] * self.envs.num_envs
        self.got_new_plan = [False] * self.envs.num_envs
        self.replan_retries = [0] * self.envs.num_envs
        self.replan = [True] * self.envs.num_envs
        self.astar_img = [None] * self.envs.num_envs

        self.first_step = True

        # data generation loop
        while not self.is_done():

            # step all envs
            self._step(self.envs)

            not_ready = False
            # if no observations or no disagr map, exit
            if not self.current_infos or not self.current_observations:
                print("\n\n\nINFOS OR OBS NOT READY")
                not_ready = True

            # for each env, visualize and send next subgoal if needed
            for idx in range(self.envs.num_envs):
                obs = self.current_observations[idx]
                info = self.current_infos[idx]
                episode = self.envs.current_episodes()[idx]

                # visualize maps for debugging purposes
                if self.visualize and self.astar_img[idx] is not None:
                    cv2.imshow("Env " + str(idx) + " A*", self.astar_img[idx])

                if self.visualize and ('disagreement_map' in obs[0]):
                    rgb_map = maps.colorize_draw_agent_and_fit_to_height(
                        info["top_down_map"], obs[0]['disagreement_map'].shape[0]
                    )
                    cv2.imshow("Env " + str(idx) + " map", rgb_map)
                    cv2.imshow(
                        "Env " + str(idx) + " disagreement", obs[0]['disagreement_map']
                    )

                # if time to go to next subgoal, do it
                self.goto_next_subgoal(idx)

            if self.visualize:
                cv2.waitKey(10)

            # if time to predict new goal
            self.compute_new_goals()
            
            for idx in range(self.envs.num_envs):
                obs = self.current_observations[idx]
                obs[0]['goals'] = self.sub_goals[idx]
                episode = self.envs.current_episodes()[idx]
                n_step = self.envs.call_at(idx, "get_step")
                paths = save_obs(self.exp_path, episode.episode_id, obs, n_step)
                generated_observations_paths.append(paths)

            for idx in range(self.envs.num_envs):
                done = self.current_dones[idx]
                self.num_steps_done += 1
                if done:
                    self.current_steps[idx] = 0

            if self.num_steps_done % 10 == 0:
                print(f"Exploration at {int(self.percent_done() * 100)}%")
        del self.current_dones
        del self.current_observations

        gc.collect()
        self.envs.close()
        return sorted(generated_observations_paths)
