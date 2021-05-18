# type: ignore
import cv2
import math
import numpy as np
import os
import time
from typing import *

from habitat.config.default import get_config
from habitat.tasks.nav.nav import NavigationGoal, NavigationTask
from habitat_baselines.agents.simple_agents import GoalFollower
from habitat_baselines.common.baseline_registry import baseline_registry

from experimenting_env.utils.sensors_utils import save_obs

from ..baselines import Baseline


@baseline_registry.register_trainer(name="frontierbaseline-v1")
class FrontierBaselinev1(Baseline):
    def __init__(self, config, exp_base_dir, **kwargs):
        super().__init__(config, exp_base_dir, GoalFollower, **kwargs)

    def compute_distance(self, a, b):
        return np.linalg.norm(b - a)

    def train(self):
        pass

    def _init_train(self):
        envs = super()._init_train()

        num_envs = envs.num_envs
        self.counter = 0
        self.visualize = False
        self.counter = [0] * num_envs
        self.sub_goals = [[] for _ in range(num_envs)]
        self.sub_goals_to_draw = [None] * num_envs
        self.sub_goals_counter = [0] * num_envs
        self.replan = [True] * num_envs
        self.replan_retries = [0] * num_envs
        self.got_new_plan = [False] * num_envs
        return envs

    def generate(self) -> None:

        envs = self._init_train()

        num_envs = envs.num_envs

        goal_to_draw = [0, 0]
        generated_observations_paths = []
        map_scale = 1.0


        while not self.is_done():

            self._step(envs)
            
            for idx in range(num_envs):
                obs = self.current_observations[idx]
                episode = envs.current_episodes()[idx]
                n_step = envs.call_at(idx, "get_step")
                self.counter[idx] += 1

                info = self.current_infos[idx]
                done = self.current_dones[idx]
                if done and n_step < 200:
                    print(f"Episode {episode} DONE before time?")
                episode = envs.current_episodes()[idx]

                if len(self.sub_goals[idx]) == 0:
                    self.replan[idx] = True

                # is it time to go to next subgoal (subgoals are already in m here)?

                if (
                    self.got_new_plan[idx] or self.current_steps[idx] % 20 == 0
                ) and len(self.sub_goals[idx]) > 0:

                    new_sub_goal = self.sub_goals[idx].pop(-1)
                    self.got_new_plan[idx] = False
                    print(
                        "picking next subgoal. Remaining subgoals=",
                        len(self.sub_goals[idx]),
                    )
                    envs.call_at(
                        idx,
                        "set_goals",
                        {"data": [NavigationGoal(position=new_sub_goal)]},
                    )


                if (
                    info is not None
                    and obs is not None
                    and self.replan[idx]
                    and self.counter[idx] > 20
                ):  # (info['distance_to_goal'] < 0.5 or self.counter[idx] > 30): # time to replan
                    # print("Replanning...")
                    # self.counter[idx] = 0
                    self.replan[idx] = False
                    self.replan_retries[idx] = 0

                    # self.counter[idx] = 0

                    print("Replanning....")
                    # allo_map = maps.get_top down_map_from_sim(self.sim, 1024, True, 0.05, 0)
                    allo_map = info['top_down_map']['map'].copy()
                    allo_map[allo_map > 1] = 1
                    expl_map = info['top_down_map']['fog_of_war_mask'].copy()

                    lower_bound, upper_bound = envs.call_at(idx, "get_map_bounds")

                    new_size = (
                        int(abs(upper_bound[0] - lower_bound[0]) / 0.025),
                        int(abs(upper_bound[2] - lower_bound[2]) / 0.025),
                    )

                    scaled_down_allo_map = cv2.resize(allo_map, new_size)
                    scaled_down_expl_map = cv2.resize(expl_map, new_size)

                    frontier_map = (
                        scaled_down_allo_map * 127 + scaled_down_expl_map * 127
                    )

                    # compute frontiers
                    edges = cv2.Canny(frontier_map, 100, 200)
                    frontiers = np.copy(edges)
                    frontier_goals = []
                    frontier_goals_info_gains = []
                    SEARCH_SIZE = 3  # 3
                    SEARCH_STRIDE = 5  # 5
                    for i in range(0, edges.shape[0] - SEARCH_STRIDE, SEARCH_STRIDE):
                        for j in range(
                            0, edges.shape[1] - SEARCH_STRIDE, SEARCH_STRIDE
                        ):
                            if edges[i, j] == 255:
                                obstacle_cells_nearby = 0
                                unknown_cells_nearby = 0
                                for k in range(-SEARCH_SIZE, SEARCH_SIZE + 1):
                                    for l in range(-SEARCH_SIZE, SEARCH_SIZE + 1):
                                        if k != 0 and l != 0:
                                            if frontier_map[i + k, j + l] == 0:
                                                obstacle_cells_nearby += 1
                                        if frontier_map[i + k, j + l] == 127:
                                            unknown_cells_nearby += 1

                                if obstacle_cells_nearby < 2:
                                    if self.visualize:
                                        cv2.circle(frontier_map, (j, i), 8, 0, -1)
                                        cv2.circle(frontier_map, (j, i), 8, 255, 1)
                                    frontier_goals.append(
                                        (j, i)
                                    )  # goals in x,y cv2 coords
                                    frontier_goals_info_gains.append(
                                        unknown_cells_nearby
                                    )

                    # select next exploration goal (greedy)
                    if len(frontier_goals) > 0:
                        print("frontiers found")

                        goal_dist = 0
                        agent_pos = [
                            obs[0]['position']['position'][2],
                            obs[0]['position']['position'][0],
                        ]
                        new_goal = frontier_goals[0]

                        # do not select a goal that is too close
                        worst_info_gain = 0
                        frontier_counter = 0
                        # select goals farther than 1m and with higher info gain, if possible
                        while (  # goal_dist < 1.0 and
                            frontier_counter < len(frontier_goals) - 1
                        ):
                            frontier_counter += 1
                            n = np.random.randint(len(frontier_goals))
                            new_goal = frontier_goals[n]
                            info_gain = frontier_goals_info_gains[n]
                            if info_gain < worst_info_gain:
                                continue
                            worst_info_gain = info_gain

                        # convert pixels to global map coords
                        grid_size = (
                            abs(upper_bound[2] - lower_bound[2])
                            / frontier_map.shape[0],
                            abs(upper_bound[0] - lower_bound[0])
                            / frontier_map.shape[1],
                        )
                        # lower_bounds are inverted, why?!
                        realworld_x = lower_bound[0] + new_goal[0] * grid_size[0]
                        realworld_y = lower_bound[2] + new_goal[1] * grid_size[1]
                        goal_world = [
                            realworld_x * map_scale,
                            realworld_y * map_scale,
                        ]  # goal in world cords (m)

                        goal_dist = np.linalg.norm(
                            np.array(agent_pos) - np.array(goal_world)
                        )

                        if self.visualize:
                            cv2.circle(
                                frontier_map,
                                (new_goal[0] + 2, new_goal[1] + 2),
                                5,
                                255,
                                -1,
                            )

                        if self.visualize:
                            cv2.imshow("frontier_map", frontier_map)

                        ####### Planning ######
                        if self.replan_retries[idx] < 3:
                            print("trying to plan #", self.replan_retries[idx])

                            self.replan_retries[idx] += 1
                            agent_pos = [
                                obs[0]['position']['position'][2],
                                obs[0]['position']['position'][0],
                            ]
                            grid_x = int((agent_pos[1] - lower_bound[0]) / grid_size[0])
                            grid_y = int((agent_pos[0] - lower_bound[2]) / grid_size[1])
                            start_pos = [grid_x, grid_y]
                            goal_pos = [
                                new_goal[0],
                                new_goal[1],
                            ]  # in pixels, cv2 cords

                            scale = 8
                            draw_scale = 2
                            scaled_down_allo_map = scaled_down_allo_map * 255

                            # actual A*

                            path, img = do_plan(
                                scaled_down_allo_map, start_pos, goal_pos, False, False
                            )

                            path.insert(0, goal_pos)
                            if self.visualize:
                                cv2.imshow("astar_img", img)

                            # extract sub-goals
                            self.sub_goals[idx] = []
                            for subgoal in path:
                                realworld_subgoal_x = lower_bound[0] + subgoal[0] * (
                                    abs(upper_bound[0] - lower_bound[0])
                                    / scaled_down_allo_map.shape[1]
                                )
                                realworld_subgoal_y = lower_bound[2] + subgoal[1] * (
                                    abs(upper_bound[2] - lower_bound[2])
                                    / scaled_down_allo_map.shape[0]
                                )
                                realworld_subgoal = [
                                    realworld_subgoal_x,
                                    obs[0]['position']['position'][1],
                                    realworld_subgoal_y,
                                ]

                                self.sub_goals[idx].append(realworld_subgoal)

                            self.sub_goals[idx].pop(-1)

                            if len(self.sub_goals[idx]) > 0:
                                self.got_new_plan[idx] = True
                            else:
                                print("A* failed, no waypoints")

                        else:  # if A* failed
                            print("A* failed...")
                            self.replan_retries[idx] = 0
                            envs.call_at(
                                idx,
                                "set_goals",
                                {"data": [NavigationGoal(position=[0, 0, 0])]},
                            )

                        ####### end A* ######

                    else:  # no new frontiers, reset episode
                        print("No new frontiers, defaulting to goal in 0,0,0")
                        envs.call_at(
                            idx,
                            "set_goals",
                            {"data": [NavigationGoal(position=[0, 0, 0])]},
                        )
                        # envs.call_at(idx, "set_done", {"done": True})

                if self.visualize:
                    cv2.waitKey(5)

                # save obs
                for idx in range(num_envs):
                    obs = self.last_observations[idx]
                    obs[0]['goals'] = self.sub_goals[idx]
                    obs[0]['actions'] = self.last_actions[idx]
                    
                    episode = envs.current_episodes()[idx]
                    n_step = envs.call_at(idx, "get_step")
                    generated_observations_paths.append(
                        save_obs(self.exp_path, episode.episode_id, obs, n_step)
                    )
                    
                self.num_steps_done += 1
                if done:
                    self.current_steps[idx] = 0

        envs.close()
        return sorted(generated_observations_paths)

        
@baseline_registry.register_trainer(name="frontierbaseline-v2")
class FrontierBaselinev2(Baseline):
    def __init__(self, config, exp_base_dir, **kwargs):
        super().__init__(config, exp_base_dir, GoalFollower, **kwargs)

    def compute_distance(self, a, b):
        return np.linalg.norm(b - a)

    def train(self):
        pass

    def _init_train(self):
        envs = super()._init_train()
        # init parameters for generate
        self.visualize = False
        self.replanning_time = [True] * envs.num_envs
        self.sub_goals = [[] for _ in range(envs.num_envs)]
        self.goal = [None] * envs.num_envs
        self.got_new_plan = [False] * envs.num_envs
        self.frontier_maps = [None] * envs.num_envs
        return envs

    # return frontier points and associated information gain
    def compute_frontiers(self, obs, info, idx):
        allocentric_map = info['top_down_map']['map'].copy()
        allocentric_map[allocentric_map > 1] = 1
        exploration_map = info['top_down_map']['fog_of_war_mask'].copy()

        lower_bound, upper_bound = self.envs.call_at(idx, "get_map_bounds")

        frontier_map = (
            allocentric_map * 127 + exploration_map * 127
        )

        # compute frontiers
        frontier_goals = []
        frontier_goals_info_gains = []

        edges = cv2.Canny(frontier_map, 100, 200)
        search_size, search_stride = 6, 10

        for i in range(0, edges.shape[0] - search_stride, search_stride):
            for j in range(
                0, edges.shape[1] - search_stride, search_stride
            ):
                if edges[i, j] == 255:
                    obstacle_cells_nearby = 0
                    unknown_cells_nearby = 0
                    for k in range(-search_size, search_size + 1):
                        for l in range(-search_size, search_size + 1):
                            if k != 0 and l != 0:
                                if frontier_map[i + k, j + l] == 0:
                                    obstacle_cells_nearby += 1
                            if frontier_map[i + k, j + l] == 127:
                                unknown_cells_nearby += 1

                    if obstacle_cells_nearby < 2:
                        frontier_goals.append(
                            (j, i)
                        )  # frontier goals are in (x,y) cv2 coords
                        frontier_goals_info_gains.append(
                            unknown_cells_nearby
                        )  # trivial information gain = unknown neighbour cells

        # save frontier image for debugging
        self.frontier_maps[idx] = cv2.cvtColor(frontier_map, cv2.COLOR_GRAY2BGR)
        for frontier in frontier_goals:
            cv2.circle(self.frontier_maps[idx], frontier, 20, (0,0,255), 4)

        return frontier_goals, frontier_goals_info_gains

    # visualize frontier map, goal and subgoals for debugging
    def visualize_map(self, obs, info, idx, downsize=0.5):
        frontier_map_rgb = self.frontier_maps[idx]
        if frontier_map_rgb is None:
            return

        agent_position = (
            obs[0]['position']['position'][2],
            obs[0]['position']['position'][0],
        )
        agent_position_pixels = self.coords2pixels(agent_position, info, idx)
        cv2.circle(frontier_map_rgb, tuple(agent_position_pixels), 14, (0,255,0), -1)

        if self.goal[idx] is not None:
            goal = [self.goal[idx][2], self.goal[idx][0]] 
            goal_pixels = self.coords2pixels(goal, info, idx)
            cv2.circle(frontier_map_rgb, tuple(goal_pixels), 20, (0,0,255), -1)

            counter = 0
            for sub_goal, next_sub_goal in zip(self.sub_goals[idx][:-1], self.sub_goals[idx][1:]):
                sub_goal = [sub_goal[2], sub_goal[0]]
                next_sub_goal = [next_sub_goal[2], next_sub_goal[0]]
                sub_goal_pixels = self.coords2pixels(sub_goal, info, idx)
                next_sub_goal_pixels = self.coords2pixels(next_sub_goal, info, idx)

                cv2.circle(frontier_map_rgb, tuple(next_sub_goal_pixels), 14, (255,counter,counter), -1)            
                cv2.line(frontier_map_rgb, tuple(sub_goal_pixels), tuple(next_sub_goal_pixels), (255,counter,counter), 4)
                counter += 80



        frontier_map_rgb = cv2.resize(frontier_map_rgb, (int(frontier_map_rgb.shape[1] * downsize), 
            int(frontier_map_rgb.shape[0] * downsize)))
        # cv2.imshow("frontier_map", frontier_map_rgb)
        # cv2.imshow("rgb", obs[0]["rgb"])
        # cv2.waitKey(5)

    # select next frontier among frontier_goals based on some heuristic (greedy for now)
    def select_next_frontier(self, obs, frontier_goals, frontier_goals_info_gains):
        new_goal = frontier_goals[0]
        worst_info_gain = 0
        frontier_counter = 0

        # what is this?!
        while (frontier_counter < len(frontier_goals) - 1):
            frontier_counter += 1
            n = np.random.randint(len(frontier_goals))
            new_goal = frontier_goals[n]
            new_info_gain = frontier_goals_info_gains[n]
            if new_info_gain < worst_info_gain:
                continue
            worst_info_gain = new_info_gain

        return new_goal

    # convert point in cv2 pixel coords to habitat 3D coords in (m)
    def pixels2coords(self, pixels, info, idx):
        lower_bound, upper_bound = self.envs.call_at(idx, "get_map_bounds")
        grid_size = (
            abs(upper_bound[2] - lower_bound[2])
            / info['top_down_map']['map'].shape[0],
            abs(upper_bound[0] - lower_bound[0])
            / info['top_down_map']['map'].shape[1],
        )
        # lower_bounds are inverted, why?!
        realworld_x = lower_bound[0] + pixels[0] * grid_size[0] # goal in world cords (m)
        realworld_y = lower_bound[2] + pixels[1] * grid_size[1]
        
        return np.array([realworld_x, 0.0, realworld_y])

    # convert habitat 3D coords in (m) to cv2 pixel coords
    def coords2pixels(self, coords, info, idx):
        lower_bound, upper_bound = self.envs.call_at(idx, "get_map_bounds")
        grid_size = (
            abs(upper_bound[2] - lower_bound[2])
            / info['top_down_map']['map'].shape[0],
            abs(upper_bound[0] - lower_bound[0])
            / info['top_down_map']['map'].shape[1],
        )
        grid_x = int((coords[1] - lower_bound[0]) / grid_size[0])
        grid_y = int((coords[0] - lower_bound[2]) / grid_size[1])

        return [grid_x, grid_y]

    def generate(self) -> None:
        self.envs = self._init_train()
        generated_observations_paths = []

        while not self.is_done():
            self._step(self.envs)

            for idx in range(self.envs.num_envs):
                obs = self.current_observations[idx]
                info = self.current_infos[idx]
                done = self.current_dones[idx]
                episode = self.envs.current_episodes()[idx]

                if done:
                    self.replanning_time = [True] * self.envs.num_envs
                    self.sub_goals = [[] for _ in range(self.envs.num_envs)]
                    self.goal = [None] * self.envs.num_envs
                    self.previous_sub_goal = [None] * self.envs.num_envs
                    self.elapsed_since_last_sub_goal = [0] * self.envs.num_envs        
                    self.got_new_plan = [False] * self.envs.num_envs
                    self.frontier_maps = [None] * self.envs.num_envs    

                if self.visualize:
                    self.visualize_map(obs, info, idx)

                # save obs
                n_step = self.envs.call_at(idx, "get_step")
                generated_observations_paths.append(
                    save_obs(self.exp_path, episode.episode_id, obs, n_step)
                )

                # TODO: check if last frontier has been reached, then replan
                # if self.num_steps_done % 10 == 0:
                #     self.replanning_time[idx] = True
                if self.num_steps_done % 20 == 0 and len(self.sub_goals[idx]) == 0:
                    self.replanning_time[idx] = True

                # check if time to go to next subgoal (either a new goal is available, or 20 steps elapsed)
                if (self.got_new_plan[idx] or self.num_steps_done % 20 == 0) and len(self.sub_goals[idx]) > 0:
                    new_sub_goal = self.sub_goals[idx].pop(0)
                    self.got_new_plan[idx] = False
                    print(
                        "picking next subgoal. Remaining subgoals=",
                        len(self.sub_goals[idx]),
                    )
                    self.envs.call_at(
                        idx,
                        "set_goals",
                        {"data": [NavigationGoal(position=new_sub_goal)]},
                    )

                # if it's time to replan
                if self.replanning_time[idx]:    
                    self.replanning_time[idx] = False

                    frontier_goals, frontier_goals_info_gains = self.compute_frontiers(obs, info, idx)

                    # greedily select next exploration goal among frontier goals and compute subgoals using pathfinder
                    if len(frontier_goals) > 0:
                        agent_position = obs[0]['position']['position']

                        new_goal_pixels = self.select_next_frontier(obs, frontier_goals, frontier_goals_info_gains)
                        new_goal_coords = self.pixels2coords(new_goal_pixels, info, idx)
                        new_goal_coords[1] = agent_position[1]

                        self.goal[idx] = new_goal_coords

                        print("New frontier goal at:", new_goal_coords, "- sending agent")

                        path_points = self.envs.call_at(
                            idx,
                            "get_path",
                            {
                                "agent_position": agent_position,
                                "goal": new_goal_coords
                            },
                        )
                        # print("Path:", path_points)
                        self.sub_goals[idx] = path_points.tolist()
                        self.got_new_plan[idx] = True # signal that a new plan is available 

                    else:  # if no new frontiers found, map is fully explored -> reset episode
                        print("No new frontiers, sending agent back to origin (0, 0)")
                        self.envs.call_at(
                            idx,
                            "set_goals",
                            {"data": [NavigationGoal(position=[0, obs[0]['position']['position'][1], 0])]},
                        )
                        self.got_new_plan[idx] = False

                self.num_steps_done += 1

        self.envs.close()

        return sorted(generated_observations_paths)


# similar to frontierbaseline-v2, but new subgoals are sent upon reaching the previous one 
@baseline_registry.register_trainer(name="frontierbaseline-v3")
class FrontierBaselinev3(Baseline):
    def __init__(self, config, exp_base_dir, **kwargs):
        super().__init__(config, exp_base_dir, GoalFollower, **kwargs)

    def compute_distance(self, a, b):
        return np.linalg.norm(b - a)

    def train(self):
        pass

    def _init_train(self):
        envs = super()._init_train()
        # init parameters for generate
        self.visualize = True
        self.replanning_time = [True] * envs.num_envs
        self.sub_goals = [[] for _ in range(envs.num_envs)]
        self.goal = [None] * envs.num_envs
        self.previous_sub_goal = [None] * envs.num_envs
        self.elapsed_since_last_sub_goal = [0] * envs.num_envs        
        self.got_new_plan = [False] * envs.num_envs
        self.frontier_maps = [None] * envs.num_envs
        self.edge_maps = [None] * envs.num_envs
        # self.episode_elapsed_steps = [0] * envs.num_envs 

        return envs

    # return frontier points and associated information gain
    def compute_frontiers(self, obs, info, idx):
        allocentric_map = info['top_down_map']['map'].copy()
        allocentric_map[allocentric_map > 1] = 1
        exploration_map = info['top_down_map']['fog_of_war_mask'].copy()

        lower_bound, upper_bound = self.envs.call_at(idx, "get_map_bounds")

        frontier_map = (
            allocentric_map * 127 + exploration_map * 127
        )

        # save frontier image for debugging
        self.frontier_maps[idx] = cv2.cvtColor(frontier_map, cv2.COLOR_GRAY2BGR)
        
        # compute frontiers
        frontier_goals = []
        frontier_goals_info_gains = []

        edges = cv2.Canny(frontier_map, 100, 200)
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel)
        self.edge_maps[idx] = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        search_size, search_stride = 20, 5

        for i in range(search_size, edges.shape[0] - search_size, search_stride):
            for j in range(
                search_size, edges.shape[1] - search_size, search_stride
            ):
                # cv2.circle(self.frontier_maps[idx], (j,i), 1, (0,255,0), -1)

                if edges[i, j] == 255:
                    obstacle_cells_nearby = 0
                    unknown_cells_nearby = 0
                    free_cells_nearby = 0
                    for k in range(-search_size, search_size + 1):
                        for l in range(-search_size, search_size + 1):
                            # if k != 0 and l != 0:
                            if frontier_map[i + k, j + l] == 0:
                                obstacle_cells_nearby += 1
                            if frontier_map[i + k, j + l] == 127:
                                unknown_cells_nearby += 1
                            if frontier_map[i + k, j + l] == 255:
                                free_cells_nearby += 1

                    if obstacle_cells_nearby < 3: #> 2 and unknown_cells_nearby > 2:
                        frontier_goals.append(
                            (j, i)
                        )  # frontier goals are in (x,y) cv2 coords
                        frontier_goals_info_gains.append(
                            unknown_cells_nearby
                        )  # trivial information gain = unknown neighbour cells

        
        for frontier in frontier_goals:
            cv2.circle(self.frontier_maps[idx], frontier, 20, (0,0,255), 4)

        return frontier_goals, frontier_goals_info_gains

    # visualize frontier map, goal and subgoals for debugging
    def visualize_map(self, obs, info, idx, downsize=0.5):
        frontier_map_rgb = self.frontier_maps[idx]
        if frontier_map_rgb is None:
            return

        agent_position = (
            obs[0]['position']['position'][2],
            obs[0]['position']['position'][0],
        )
        agent_position_pixels = self.coords2pixels(agent_position, info, idx)
        cv2.circle(frontier_map_rgb, tuple(agent_position_pixels), 14, (0,255,0), -1)

        if self.goal[idx] is not None:
            goal = [self.goal[idx][2], self.goal[idx][0]] 
            goal_pixels = self.coords2pixels(goal, info, idx)
            cv2.circle(frontier_map_rgb, tuple(goal_pixels), 20, (0,0,255), -1)

            counter = 0
            for sub_goal, next_sub_goal in zip(self.sub_goals[idx][:-1], self.sub_goals[idx][1:]):
                sub_goal = [sub_goal[2], sub_goal[0]]
                next_sub_goal = [next_sub_goal[2], next_sub_goal[0]]
                sub_goal_pixels = self.coords2pixels(sub_goal, info, idx)
                next_sub_goal_pixels = self.coords2pixels(next_sub_goal, info, idx)

                cv2.circle(frontier_map_rgb, tuple(next_sub_goal_pixels), 14, (255,counter,counter), -1)            
                cv2.line(frontier_map_rgb, tuple(sub_goal_pixels), tuple(next_sub_goal_pixels), (255,counter,counter), 4)
                counter += 80

        frontier_map_rgb = cv2.resize(frontier_map_rgb, (int(frontier_map_rgb.shape[1] * downsize), 
            int(frontier_map_rgb.shape[0] * downsize)))
#        cv2.imshow("frontier_map", frontier_map_rgb)

        edge_map_rgb = self.edge_maps[idx]
        edge_map_rgb = cv2.resize(edge_map_rgb, (int(edge_map_rgb.shape[1] * downsize), 
            int(edge_map_rgb.shape[0] * downsize)))        
        # cv2.imshow("edge_map", edge_map_rgb)

        # cv2.imshow("rgb", obs[0]["rgb"])
        # cv2.waitKey(5)

    # select next frontier among frontier_goals based on some heuristic (greedy for now)
    def select_next_frontier(self, obs, frontier_goals, frontier_goals_info_gains):
        new_goal = frontier_goals[0]
        worst_info_gain = 0
        frontier_counter = 0

        # what is this?!
        while (frontier_counter < len(frontier_goals) - 1):
            frontier_counter += 1
            n = np.random.randint(len(frontier_goals))
            new_goal = frontier_goals[n]
            new_info_gain = frontier_goals_info_gains[n]
            if new_info_gain < worst_info_gain:
                continue
            worst_info_gain = new_info_gain

        return new_goal

    # convert point in cv2 pixel coords to habitat 3D coords in (m)
    def pixels2coords(self, pixels, info, idx):
        lower_bound, upper_bound = self.envs.call_at(idx, "get_map_bounds")
        grid_size = (
            abs(upper_bound[2] - lower_bound[2])
            / info['top_down_map']['map'].shape[0],
            abs(upper_bound[0] - lower_bound[0])
            / info['top_down_map']['map'].shape[1],
        )
        # lower_bounds are inverted, why?!
        realworld_x = lower_bound[0] + pixels[0] * grid_size[0] # goal in world cords (m)
        realworld_y = lower_bound[2] + pixels[1] * grid_size[1]
        
        return np.array([realworld_x, 0.0, realworld_y])

    # convert habitat 3D coords in (m) to cv2 pixel coords
    def coords2pixels(self, coords, info, idx):
        lower_bound, upper_bound = self.envs.call_at(idx, "get_map_bounds")
        grid_size = (
            abs(upper_bound[2] - lower_bound[2])
            / info['top_down_map']['map'].shape[0],
            abs(upper_bound[0] - lower_bound[0])
            / info['top_down_map']['map'].shape[1],
        )
        grid_x = int((coords[1] - lower_bound[0]) / grid_size[0])
        grid_y = int((coords[0] - lower_bound[2]) / grid_size[1])

        return [grid_x, grid_y]

    def generate(self) -> None:
        self.envs = self._init_train()
        generated_observations_paths = []

        while not self.is_done():
            if self.num_steps_done % self.config.ENVIRONMENT.MAX_EPISODE_STEPS == 0 or \
                self.num_steps_done % self.config.ENVIRONMENT.MAX_EPISODE_STEPS > 20:
                    self._step(self.envs)

            for idx in range(self.envs.num_envs):
                obs = self.current_observations[idx]
                info = self.current_infos[idx]
                done = self.current_dones[idx]
                episode = self.envs.current_episodes()[idx]

                # self.envs.call_at(
                #     idx,
                #     "save_to_images",
                #     {"observations": obs, "info": info},
                # )

                # print("step:", self.num_steps_done, "done:", done, "episode:", episode.episode_id)
                if done:
                    self.replanning_time = [True] * self.envs.num_envs
                    self.sub_goals = [[] for _ in range(self.envs.num_envs)]
                    self.goal = [None] * self.envs.num_envs
                    self.previous_sub_goal = [None] * self.envs.num_envs
                    self.elapsed_since_last_sub_goal = [0] * self.envs.num_envs        
                    self.got_new_plan = [False] * self.envs.num_envs
                    self.frontier_maps = [None] * self.envs.num_envs 
                    # self.episode_elapsed_steps = [0] * self.envs.num_envs 
                                  
                if self.visualize:
                    self.visualize_map(obs, info, idx)

                # save obs
                n_step = self.envs.call_at(idx, "get_step")
                generated_observations_paths.append(
                    save_obs(self.exp_path, episode.episode_id, obs, n_step)
                )

                # for the first 10 steps, rotate in place
                if self.num_steps_done % self.config.ENVIRONMENT.MAX_EPISODE_STEPS < 21:
                    self.envs.call_at(
                        idx,
                        "step",
                        {"action": 2},
                    )
                    print("Rotating in place...")
                    self.num_steps_done += 1
                    continue

                self.elapsed_since_last_sub_goal[idx] += 1

                # check distance between agent and last subgoal
                previous_subgoal_reached = False
                if self.previous_sub_goal[idx] is not None:
                    distance_to_previous_subgoal = np.linalg.norm(
                                                        self.previous_sub_goal[idx] - 
                                                        obs[0]['position']['position']
                                                        )
                    if distance_to_previous_subgoal < 0.25:
                        previous_subgoal_reached = True

                        # TODO: check if last frontier has been reached, then replan
                        if len(self.sub_goals[idx]) == 0:
                            self.replanning_time[idx] = True
                            self.previous_sub_goal[idx] = None

                # check if time to go to next subgoal (either a new goal is available, or 20 steps elapsed)
                if len(self.sub_goals[idx]) > 0 and (previous_subgoal_reached or self.got_new_plan[idx] or self.elapsed_since_last_sub_goal[idx] > 40):
                    new_sub_goal = self.sub_goals[idx].pop(0)
                    self.got_new_plan[idx] = False
                    print(
                        "picking next subgoal. Remaining subgoals=",
                        len(self.sub_goals[idx]),
                    )
                    self.envs.call_at(
                        idx,
                        "set_goals",
                        {"data": [NavigationGoal(position=new_sub_goal)]},
                    )
                    self.previous_sub_goal[idx] = new_sub_goal
                    self.elapsed_since_last_sub_goal[idx] = 0


                # if it's time to replan
                if self.replanning_time[idx]:    
                    self.replanning_time[idx] = False

                    frontier_goals, frontier_goals_info_gains = self.compute_frontiers(obs, info, idx)

                    # greedily select next exploration goal among frontier goals and compute subgoals using pathfinder
                    if len(frontier_goals) > 0:
                        agent_position = obs[0]['position']['position']

                        new_goal_pixels = self.select_next_frontier(obs, frontier_goals, frontier_goals_info_gains)
                        new_goal_coords = self.pixels2coords(new_goal_pixels, info, idx)
                        new_goal_coords[1] = agent_position[1]

                        self.goal[idx] = new_goal_coords

                        print("New frontier goal at:", new_goal_coords, "- sending agent")

                        path_points = self.envs.call_at(
                            idx,
                            "get_path",
                            {
                                "agent_position": agent_position,
                                "goal": new_goal_coords
                            },
                        )
                        # print("Path:", path_points)
                        self.sub_goals[idx] = path_points.tolist()
                        self.got_new_plan[idx] = True # signal that a new plan is available 

                    else:  # if no new frontiers found, map is fully explored -> reset episode
                        print("No new frontiers, sending agent back to origin (0, 0)")
                        self.envs.call_at(
                            idx,
                            "set_goals",
                            {"data": [NavigationGoal(position=[0, obs[0]['position']['position'][1], 0])]},
                        )
                        self.got_new_plan[idx] = False

                self.num_steps_done += 1

        self.envs.close()

        return sorted(generated_observations_paths)
