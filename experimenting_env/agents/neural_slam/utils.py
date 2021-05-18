import pickle

import attr
import habitat
import habitat_sim
import habitat_sim.utils
import magnum as mn
import numpy as np

# actuation_noise_fwd = pickle.load(open("noise_models/actuation_noise_fwd.pkl", 'rb'))
# actuation_noise_right = pickle.load(
#     open("noise_models/actuation_noise_right.pkl", 'rb')
# )
# actuation_noise_left = pickle.load(open("noise_models/actuation_noise_left.pkl", 'rb'))


@attr.s(auto_attribs=True, slots=True)
class CustomActuationSpec:
    action: int


def _custom_action_impl(
    scene_node: habitat_sim.SceneNode,
    delta_dist: float,  # in metres
    delta_dist_angle: float,  # in degrees
    delta_angle: float,  # in degrees
):
    forward_ax = (
        np.array(scene_node.absolute_transformation().rotation_scaling())
        @ habitat_sim.geo.FRONT
    )
    move_angle = np.deg2rad(delta_dist_angle)

    rotation = habitat_sim.utils.quat_from_angle_axis(move_angle, habitat_sim.geo.UP)
    move_ax = habitat_sim.utils.quat_rotate_vector(rotation, forward_ax)

    scene_node.translate_local(move_ax * delta_dist)
    scene_node.rotate_local(mn.Deg(delta_angle), habitat_sim.geo.UP)


def _noisy_action_impl(scene_node: habitat_sim.SceneNode, action: int):
    if action == 0:  ## Forward
        dx, dy, do = actuation_noise_fwd.sample()[0][0]
    elif action == 1:  ## Right
        dx, dy, do = actuation_noise_right.sample()[0][0]
    elif action == 2:  ## Left
        dx, dy, do = actuation_noise_left.sample()[0][0]

    delta_dist = np.sqrt(dx ** 2 + dy ** 2)
    delta_dist_angle = np.rad2deg(np.arctan2(-dy, dx))
    delta_angle = -do

    _custom_action_impl(scene_node, delta_dist, delta_dist_angle, delta_angle)


# from habitat_sim.agent.controls import register_move_f
# @register_move_fn(body_action=True)
# class NoisyForward(habitat_sim.SceneNodeControl):
#     def __call__(
#         self,
#         scene_node: habitat_sim.SceneNode,
#         actuation_spec: CustomActuationSpec,
#     ):
#         _noisy_action_impl(
#             scene_node,
#             actuation_spec.action,
#         )


# @register_move_fn(body_action=True)
# class NoisyRight(habitat_sim.SceneNodeControl):
#     def __call__(
#         self,
#         scene_node: habitat_sim.SceneNode,
#         actuation_spec: CustomActuationSpec,
#     ):
#         _noisy_action_impl(
#             scene_node,
#             actuation_spec.action,
#         )


# @register_move_fn(body_action=True)
# class NoisyLeft(habitat_sim.SceneNodeControl):
#     def __call__(
#         self,
#         scene_node: habitat_sim.SceneNode,
#         actuation_spec: CustomActuationSpec,
#     ):
#         _noisy_action_impl(
#             scene_node,
#             actuation_spec.action,
#         )


# # from habitat.sims.habitat_simulator.action_spaces import (
# #     HabitatSimV0ActionSpaceConfiguration,
# # )


# @habitat.registry.register_action_space_configuration
# class CustomActionSpaceConfiguration(HabitatSimV0ActionSpaceConfiguration):
#     def get(self):
#         config = super().get()

#         config[habitat.SimulatorActions.NOISY_FORWARD] = habitat_sim.ActionSpec(
#             "noisy_forward",
#             CustomActuationSpec(0),
#         )
#         config[habitat.SimulatorActions.NOISY_RIGHT] = habitat_sim.ActionSpec(
#             "noisy_right",
#             CustomActuationSpec(1),
#         )
#         config[habitat.SimulatorActions.NOISY_LEFT] = habitat_sim.ActionSpec(
#             "noisy_left",
#             CustomActuationSpec(2),
#         )

#         return config


def get_l2_distance(x1, x2, y1, y2):
    """
    Computes the L2 distance between two points.
    """
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def get_rel_pose_change(pos2, pos1):
    x1, y1, o1 = pos1
    x2, y2, o2 = pos2

    theta = np.arctan2(y2 - y1, x2 - x1) - o1
    dist = get_l2_distance(x1, x2, y1, y2)
    dx = dist * np.cos(theta)
    dy = dist * np.sin(theta)
    do = o2 - o1

    return dx, dy, do


def get_new_pose(pose, rel_pose_change):
    x, y, o = pose
    dx, dy, do = rel_pose_change

    global_dx = dx * np.sin(np.deg2rad(o)) + dy * np.cos(np.deg2rad(o))
    global_dy = dx * np.cos(np.deg2rad(o)) - dy * np.sin(np.deg2rad(o))
    x += global_dy
    y += global_dx
    o += np.rad2deg(do)
    if o > 180.0:
        o -= 360.0

    return x, y, o


def threshold_poses(coords, shape):
    coords[0] = min(max(0, coords[0]), shape[0] - 1)
    coords[1] = min(max(0, coords[1]), shape[1] - 1)
    return coords


import numpy as np


class HabitatMaps(object):
    def __init__(self, env, N=int(1e6), resolution=5, padding=0):
        # convert to cm
        self.resolution = resolution
        self.padding = padding

        pts = self._sample_points(env, N) * 100.0

        # Bin points based on x and z values, so that
        # we can quickly pool them based on y-filtering.
        self.y = pts[:, 1]
        zx = pts[:, [2, 0]]
        self.origin, self.size, self.max = self._make_map(
            zx, self.padding, self.resolution
        )

        zx = zx - self.origin
        self.zx = (zx / self.resolution).astype(np.int)

    def get_map(self, y, lb, ub):
        ids = np.logical_and(self.y > y + lb, self.y < y + ub)
        num_points = np.zeros((self.size[1], self.size[0]), dtype=np.int32)
        np.add.at(num_points, (self.zx[ids, 1], self.zx[ids, 0]), 1)
        return num_points

    def _make_map(self, zx, padding, resolution):
        """Returns a map structure."""
        min_, max_ = self._get_xy_bounding_box(zx, padding=padding)
        sz = np.ceil((max_ - min_ + 1) / resolution).astype(np.int32)
        max_ = min_ + sz * resolution - 1
        return min_, sz, max_

    def _get_xy_bounding_box(self, zx, padding):
        """Returns the xy bounding box of the environment."""
        min_ = np.floor(np.min(zx, axis=0) - padding).astype(np.int)
        max_ = np.ceil(np.max(zx, axis=0) + padding).astype(np.int)
        return min_, max_

    def _sample_points(self, env, N):
        pts = np.zeros((N, 3), dtype=np.float32)
        for i in range(N):
            pts[i, :] = env.sim.sample_navigable_point()
        return pts


import sys

import matplotlib
import numpy as np

if sys.platform == 'darwin':
    matplotlib.use("tkagg")
else:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import skimage


def visualize(
    fig,
    ax,
    img,
    grid,
    pos,
    gt_pos,
    dump_dir,
    rank,
    ep_no,
    t,
    visualize,
    print_images,
    vis_style,
):
    import seaborn as sns
    for i in range(2):
        ax[i].clear()
        ax[i].set_yticks([])
        ax[i].set_xticks([])
        ax[i].set_yticklabels([])
        ax[i].set_xticklabels([])

    ax[0].imshow(img)
    ax[0].set_title(
        "Observation", family='sans-serif', fontname='Helvetica', fontsize=20
    )

    if vis_style == 1:
        title = "Predicted Map and Pose"
    else:
        title = "Ground-Truth Map and Pose"

    ax[1].imshow(grid)
    ax[1].set_title(title, family='sans-serif', fontname='Helvetica', fontsize=20)

    # Draw GT agent pose
    agent_size = 8
    x, y, o = gt_pos
    x, y = x * 100.0 / 5.0, grid.shape[1] - y * 100.0 / 5.0

    dx = 0
    dy = 0
    fc = 'Grey'
    dx = np.cos(np.deg2rad(o))
    dy = -np.sin(np.deg2rad(o))
    ax[1].arrow(
        x - 1 * dx,
        y - 1 * dy,
        dx * agent_size,
        dy * (agent_size * 1.25),
        head_width=agent_size,
        head_length=agent_size * 1.25,
        length_includes_head=True,
        fc=fc,
        ec=fc,
        alpha=0.9,
    )

    # Draw predicted agent pose
    x, y, o = pos
    x, y = x * 100.0 / 5.0, grid.shape[1] - y * 100.0 / 5.0

    dx = 0
    dy = 0
    fc = 'Red'
    dx = np.cos(np.deg2rad(o))
    dy = -np.sin(np.deg2rad(o))
    ax[1].arrow(
        x - 1 * dx,
        y - 1 * dy,
        dx * agent_size,
        dy * agent_size * 1.25,
        head_width=agent_size,
        head_length=agent_size * 1.25,
        length_includes_head=True,
        fc=fc,
        ec=fc,
        alpha=0.6,
    )

    for _ in range(5):
        plt.tight_layout()

    if visualize:
        plt.gcf().canvas.flush_events()
        fig.canvas.start_event_loop(0.001)
        plt.gcf().canvas.flush_events()

    if print_images:
        fn = '{}/episodes/{}/{}/{}-{}-Vis-{}.png'.format(
            dump_dir, (rank + 1), ep_no, rank, ep_no, t
        )
        plt.savefig(fn)


def insert_circle(mat, x, y, value):
    mat[x - 2 : x + 3, y - 2 : y + 3] = value
    mat[x - 3 : x + 4, y - 1 : y + 2] = value
    mat[x - 1 : x + 2, y - 3 : y + 4] = value
    return mat


def fill_color(colored, mat, color):
    for i in range(3):
        colored[:, :, 2 - i] *= 1 - mat
        colored[:, :, 2 - i] += (1 - color[i]) * mat
    return colored


def get_colored_map(
    mat, collision_map, visited, visited_gt, goal, explored, gt_map, gt_map_explored
):
    import seaborn as sn
    m, n = mat.shape
    colored = np.zeros((m, n, 3))
    pal = sns.color_palette("Paired")

    current_palette = [(0.9, 0.9, 0.9)]
    colored = fill_color(colored, gt_map, current_palette[0])

    current_palette = [(235.0 / 255.0, 243.0 / 255.0, 1.0)]
    colored = fill_color(colored, explored, current_palette[0])

    green_palette = sns.light_palette("green")
    colored = fill_color(colored, mat, pal[2])

    current_palette = [(0.6, 0.6, 0.6)]
    colored = fill_color(colored, gt_map_explored, current_palette[0])

    colored = fill_color(colored, mat * gt_map_explored, pal[3])

    red_palette = sns.light_palette("red")

    colored = fill_color(colored, visited_gt, current_palette[0])
    colored = fill_color(colored, visited, pal[4])
    colored = fill_color(colored, visited * visited_gt, pal[5])

    colored = fill_color(colored, collision_map, pal[2])

    current_palette = sns.color_palette()

    selem = skimage.morphology.disk(4)
    goal_mat = np.zeros((m, n))
    goal_mat[goal[0], goal[1]] = 1
    goal_mat = 1 - skimage.morphology.binary_dilation(goal_mat, selem) != True

    colored = fill_color(colored, goal_mat, current_palette[0])

    current_palette = sns.color_palette("Paired")

    colored = 1 - colored
    colored *= 255
    colored = colored.astype(np.uint8)
    return colored


# The following code is largely borrowed from:
# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/storage.py

from collections import namedtuple

import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


def _flatten_helper(T, N, _tensor):
    return _tensor.view(T * N, *_tensor.size()[2:])


class RolloutStorage(object):
    def __init__(
        self, num_steps, num_processes, obs_shape, action_space, rec_state_size
    ):

        if action_space.__class__.__name__ == 'Discrete':
            self.n_actions = 1
            action_type = torch.long
        else:
            self.n_actions = action_space.shape[0]
            action_type = torch.float32

        self.obs = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.rec_states = torch.zeros(num_steps + 1, num_processes, rec_state_size)
        self.rewards = torch.zeros(num_steps, num_processes)
        self.value_preds = torch.zeros(num_steps + 1, num_processes)
        self.returns = torch.zeros(num_steps + 1, num_processes)
        self.action_log_probs = torch.zeros(num_steps, num_processes)
        self.actions = torch.zeros(
            (num_steps, num_processes, self.n_actions), dtype=action_type
        )
        self.masks = torch.ones(num_steps + 1, num_processes)

        self.num_steps = num_steps
        self.step = 0
        self.has_extras = False
        self.extras_size = None

    def to(self, device):
        self.obs = self.obs.to(device)
        self.rec_states = self.rec_states.to(device)
        self.rewards = self.rewards.to(device)
        self.value_preds = self.value_preds.to(device)
        self.returns = self.returns.to(device)
        self.action_log_probs = self.action_log_probs.to(device)
        self.actions = self.actions.to(device)
        self.masks = self.masks.to(device)
        if self.has_extras:
            self.extras = self.extras.to(device)
        return self

    def insert(
        self, obs, rec_states, actions, action_log_probs, value_preds, rewards, masks
    ):
        self.obs[self.step + 1].copy_(obs)
        self.rec_states[self.step + 1].copy_(rec_states)
        self.actions[self.step].copy_(actions.view(-1, self.n_actions))
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.num_steps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.rec_states[0].copy_(self.rec_states[-1])
        self.masks[0].copy_(self.masks[-1])
        if self.has_extras:
            self.extras[0].copy_(self.extras[-1])

    def compute_returns(self, next_value, use_gae, gamma, tau):
        if use_gae:
            self.value_preds[-1] = next_value
            gae = 0
            for step in reversed(range(self.rewards.size(0))):
                delta = (
                    self.rewards[step]
                    + gamma * self.value_preds[step + 1] * self.masks[step + 1]
                    - self.value_preds[step]
                )
                gae = delta + gamma * tau * self.masks[step + 1] * gae
                self.returns[step] = gae + self.value_preds[step]
        else:
            self.returns[-1] = next_value
            for step in reversed(range(self.rewards.size(0))):
                self.returns[step] = (
                    self.returns[step + 1] * gamma * self.masks[step + 1]
                    + self.rewards[step]
                )

    def feed_forward_generator(self, advantages, num_mini_batch):

        num_steps, num_processes = self.rewards.size()[0:2]
        batch_size = num_processes * num_steps
        mini_batch_size = batch_size // num_mini_batch
        assert batch_size >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "* number of steps ({}) = {} "
            "to be greater than or equal to the number of PPO mini batches ({})."
            "".format(
                num_processes, num_steps, num_processes * num_steps, num_mini_batch
            )
        )

        sampler = BatchSampler(
            SubsetRandomSampler(range(batch_size)), mini_batch_size, drop_last=False
        )

        for indices in sampler:
            yield {
                'obs': self.obs[:-1].view(-1, *self.obs.size()[2:])[indices],
                'rec_states': self.rec_states[:-1].view(-1, self.rec_states.size(-1))[
                    indices
                ],
                'actions': self.actions.view(-1, self.n_actions)[indices],
                'value_preds': self.value_preds[:-1].view(-1)[indices],
                'returns': self.returns[:-1].view(-1)[indices],
                'masks': self.masks[:-1].view(-1)[indices],
                'old_action_log_probs': self.action_log_probs.view(-1)[indices],
                'adv_targ': advantages.view(-1)[indices],
                'extras': self.extras[:-1].view(-1, self.extras_size)[indices]
                if self.has_extras
                else None,
            }

    def recurrent_generator(self, advantages, num_mini_batch):

        num_processes = self.rewards.size(1)
        assert num_processes >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(num_processes, num_mini_batch)
        )
        num_envs_per_batch = num_processes // num_mini_batch
        perm = torch.randperm(num_processes)
        T, N = self.num_steps, num_envs_per_batch

        for start_ind in range(0, num_processes, num_envs_per_batch):

            obs = []
            rec_states = []
            actions = []
            value_preds = []
            returns = []
            masks = []
            old_action_log_probs = []
            adv_targ = []
            if self.has_extras:
                extras = []

            for offset in range(num_envs_per_batch):

                ind = perm[start_ind + offset]
                obs.append(self.obs[:-1, ind])
                rec_states.append(self.rec_states[0:1, ind])
                actions.append(self.actions[:, ind])
                value_preds.append(self.value_preds[:-1, ind])
                returns.append(self.returns[:-1, ind])
                masks.append(self.masks[:-1, ind])
                old_action_log_probs.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])
                if self.has_extras:
                    extras.append(self.extras[:-1, ind])

            # These are all tensors of size (T, N, ...)
            obs = torch.stack(obs, 1)
            actions = torch.stack(actions, 1)
            value_preds = torch.stack(value_preds, 1)
            returns = torch.stack(returns, 1)
            masks = torch.stack(masks, 1)
            old_action_log_probs = torch.stack(old_action_log_probs, 1)
            adv_targ = torch.stack(adv_targ, 1)
            if self.has_extras:
                extras = torch.stack(extras, 1)

            yield {
                'obs': _flatten_helper(T, N, obs),
                'actions': _flatten_helper(T, N, actions),
                'value_preds': _flatten_helper(T, N, value_preds),
                'returns': _flatten_helper(T, N, returns),
                'masks': _flatten_helper(T, N, masks),
                'old_action_log_probs': _flatten_helper(T, N, old_action_log_probs),
                'adv_targ': _flatten_helper(T, N, adv_targ),
                'extras': _flatten_helper(T, N, extras) if self.has_extras else None,
                'rec_states': torch.stack(rec_states, 1).view(N, -1),
            }


class GlobalRolloutStorage(RolloutStorage):
    def __init__(
        self,
        num_steps,
        num_processes,
        obs_shape,
        action_space,
        rec_state_size,
        extras_size,
    ):
        super(GlobalRolloutStorage, self).__init__(
            num_steps, num_processes, obs_shape, action_space, rec_state_size
        )
        self.extras = torch.zeros(
            (num_steps + 1, num_processes, extras_size), dtype=torch.long
        )
        self.has_extras = True
        self.extras_size = extras_size

    def insert(
        self,
        obs,
        rec_states,
        actions,
        action_log_probs,
        value_preds,
        rewards,
        masks,
        extras,
    ):
        self.extras[self.step + 1].copy_(extras)
        super(GlobalRolloutStorage, self).insert(
            obs, rec_states, actions, action_log_probs, value_preds, rewards, masks
        )


Datapoint = namedtuple('Datapoint', ('input', 'target'))


class FIFOMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a datapoint."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = Datapoint(*args)
        if self.position == 0:
            x = self.memory[0][0]
            y = self.memory[0][1]
            self.batch_in_sizes = {}
            self.n_inputs = len(x)
            for dim in range(len(x)):
                self.batch_in_sizes[dim] = x[dim].size()

            self.batch_out_sizes = {}
            self.n_outputs = len(y)
            for dim in range(len(y)):
                self.batch_out_sizes[dim] = y[dim].size()

        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Samples a batch"""

        batch = {}
        inputs = []
        outputs = []

        for dim in range(self.n_inputs):
            inputs.append(
                torch.cat(
                    batch_size * [torch.zeros(self.batch_in_sizes[dim]).unsqueeze(0)]
                )
            )

        for dim in range(self.n_outputs):
            outputs.append(
                torch.cat(
                    batch_size * [torch.zeros(self.batch_out_sizes[dim]).unsqueeze(0)]
                )
            )

        indices = np.random.choice(len(self.memory), batch_size, replace=False)

        count = 0
        for i in indices:
            x = self.memory[i][0]
            y = self.memory[i][1]

            for dim in range(len(x)):
                inputs[dim][count] = x[dim]

            for dim in range(len(y)):
                outputs[dim][count] = y[dim]

            count += 1

        return (inputs, outputs)

    def __len__(self):
        return len(self.memory)
