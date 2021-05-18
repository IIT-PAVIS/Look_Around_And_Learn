import os
from typing import Any

import habitat
import habitat_sim
import numpy as np
from habitat import RLEnv
from habitat.core.registry import registry

from .sensors import *


class BaseEnv(RLEnv):
    def __init__(self, config, *args, **kwargs):

        super().__init__(config)
        self.config = config
        self.kwargs = kwargs
        self._previous_action = None
        self.episode_over = False
        self._elapsed_steps = 0
        self._max_episode_steps = self._env._max_episode_steps

    def reset(self):
        self._elapsed_steps = 0
        self.episode_over = False
        return super().reset()

    def set_goals(self, data):
        self.current_episode.goals = data.copy()

    def set_done(self, done):
        self.current_episode.episode_over = done

    def get_map_bounds(self):
        l, u = self.habitat_env.sim.pathfinder.get_bounds()
        return l, u

    def get_last_reward(self):
        return self.last_reward

    def get_path(self, agent_position, goal):
        pathfinder = self.habitat_env.sim.pathfinder
        if not pathfinder.is_loaded:
            print("Pathfinder not initialized, aborting.")
        else:
            path = habitat_sim.ShortestPath()
            path.requested_start = agent_position.reshape(3, 1)
            path.requested_end = pathfinder.snap_point(goal.reshape(3, 1))
            found_path = pathfinder.find_path(path)
            # print("Found path:", found_path)
        return np.array(path.points)

    def _past_limit(self) -> bool:
        if (
            self._max_episode_steps != 0
            and self._max_episode_steps <= self._elapsed_steps
        ):
            return True

        return False

    def step(self, action):
        self._elapsed_steps += 1
        self._previous_action = action
        observations = self.habitat_env.step(action)
          # single cam, but still expect to receive a list [ob1, ... obn] where i = idx of camera
        return observations
    def get_step(self):
        return self._elapsed_steps
