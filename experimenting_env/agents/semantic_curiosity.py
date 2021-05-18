# type: ignore
import math
import os
import time
from collections import deque
from typing import *
from skimage.transform import resize
import gym
from habitat.tasks.nav.nav import NavigationGoal, NavigationTask
from habitat.utils.visualizations import maps
from habitat_baselines.agents.simple_agents import GoalFollower
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.tensorboard_utils import TensorboardWriter

from experimenting_env.agents.model import *
from experimenting_env.agents.ppo import *
from experimenting_env.detector.model import MultiStageModel
from experimenting_env.sensor_data import BBSense
from experimenting_env.utils.sensors_utils import save_obs
from experimenting_env.utils.skeleton import *
from experimenting_env.utils.storage import *

from .baselines import Baseline


@baseline_registry.register_trainer(name="curiosity-v0")
class SemanticCuriosityExplorationBaseline(Baseline):
    def __init__(self, config, exp_base_dir, detectron_args, **kwargs):
        super().__init__(config, exp_base_dir, GoalFollower, **kwargs)
        self.config = config

        # reduced canonical map size for network input
        self.map_width, self.map_height = 400, 400
        self.visualize = self.config.ppo.visualize  # visualize maps

        self.object_detector = ObjectDetectorEnv(
            {"cfg": detectron_args}, device="cuda:0"
        )

    def _step(self, envs):
        (
            self.g_value,
            self.g_action,
            self.g_action_log_prob,
            self.g_rec_states,
        ) = self.g_policy.act(
            inputs=self.g_rollouts.obs[0],
            rnn_hxs=self.g_rollouts.rec_states[0],
            masks=self.g_rollouts.masks[0],
            extras=self.g_rollouts.extras[0],
            deterministic=False,
        )

        for index_env in range(envs.num_envs):
            action=self.g_action[index_env].cpu().item()       
            if action == 2:  # Forward
                action = 1
            elif action == 1:  # Right
                action = 3
                # noisy_action = habitat.SimulatorActions.NOISY_RIGHT
            elif action == 0:  # Left
                action = 2
            envs.async_step_at(index_env, action)

        results = [envs.wait_step_at(index_env) for index_env in range(envs.num_envs)]

        self.current_observations = [r[0] for r in results]
        self.current_rewards = [r[1] for r in results]
        self.current_dones = [r[2] for r in results]
        self.current_infos = [r[3] for r in results]

        self.current_steps += 1

        self.predict_current_bbs_and_update_pcd()

    def predict_current_bbs_and_update_pcd(self, detector_batch_size=8):
        # predict bbs with object detector

        for idx in range(0, self.envs.num_envs, detector_batch_size):
            max_idx = min(self.envs.num_envs, idx + detector_batch_size)
            images = [
                self.current_observations[i][0]['rgb'] for i in range(idx, max_idx)
            ]
            preds = self.object_detector.predict_batch(images)

            for i in range(max_idx - idx):
                current_env = idx + i
                # Why is it starting from 1?
                self.current_observations[current_env][0]['bbs'] = preds[0][
                    i
                ]  # dropping feature vectors


                self.envs.call_at(
                    current_env,
                    "_update_pointcloud",
                    {"observations": self.current_observations[current_env]},
                )
                disagreement_map = self.envs.call_at(
                    current_env, "get_and_update_semantic_curiosity_map"
                )
                self.current_observations[current_env][0][
                    'disagreement_map'
                ] = disagreement_map

    def get_rewards(self):

        return torch.from_numpy(
            np.array(
                [
                    self.envs.call_at(idx, "get_reward")
                    for idx in range(self.envs.num_envs)
                ]
            ).astype("int64")
        )

    def create_policy_inputs(self):
        # create disagreeemt inputs
        disagreement_inputs = [ ] 

        for idx in range(self.envs.num_envs):
            disag = self.current_observations[idx][0]['disagreement_map']
            last_dim = (self.map_height - disag.shape[-1])  
            last_dim2 = (self.map_width - disag.shape[-2])
            padded_disag = F.pad(torch.tensor(disag), (0, last_dim, 0, last_dim2))
            disagreement_inputs.append(padded_disag)
                            
        # TODO How to pad maps to get images
        global_inputs = torch.cat(disagreement_inputs, dim=1)

        return global_inputs

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

        # goal policy observation space
        self.g_observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(BBSense.CLASSES), self.map_width, self.map_height),
            dtype='uint8',
        )

        # goal policy action space
        self.g_action_space = self.envs.action_spaces[0]             

        # goal policy init
        device = torch.device("cuda:0" if self.config.ppo.cuda else "cpu")

        self.g_policy = RL_Policy(
            self.g_observation_space.shape,
            self.g_action_space,
            model_type=2,
            base_kwargs={
                'recurrent': self.config.ppo.use_recurrent_global,
                'hidden_size': self.config.ppo.g_hidden_size,
                'downscaling': self.config.ppo.global_downscaling,
            },
        ).to(device)


        # rollout storage
        self.g_rollouts = GlobalRolloutStorage(
            self.config.ppo.num_global_steps,
            self.envs.num_envs,
            self.g_observation_space.shape,
            self.g_action_space,
            self.g_policy.rec_state_size,
            1,
        ).to(device)

        # create queues
        self.g_value_losses = deque(maxlen=1000)
        self.g_action_losses = deque(maxlen=1000)
        self.g_dist_entropies = deque(maxlen=1000)

        # first forward pass

        self.global_input = torch.rand(
            self.envs.num_envs, len(BBSense.CLASSES), self.map_height, self.map_width
        )
        self.g_rollouts.obs[0].copy_(self.global_input)
        self.g_masks = torch.ones(self.envs.num_envs).float().to(device)  # not used

        torch.set_grad_enabled(False)

        if self.config.ppo.load_checkpoint:
            if os.path.exists(self.config.ppo.load_checkpoint_path):
                self.g_policy.load_state_dict(
                    torch.load(self.config.ppo.load_checkpoint_path)
                )
            else:
                print("Error: checkpoint path does not exist.")
                return

        print("Semantic curiosity init done...")

        # data generation loop
        while not self.is_done():

            # step all envs
            self._step(self.envs)
            self.global_input = self.create_policy_inputs()

            # for each env, visualize and send next subgoal if needed
            for idx in range(self.envs.num_envs):
                obs = self.current_observations[idx]
                episode = self.envs.current_episodes()[idx]
                info = self.current_infos[idx]
                episode = self.envs.current_episodes()[idx]
                n_step = self.envs.call_at(idx, "get_step")
                paths = save_obs(self.exp_path, episode.episode_id, obs, n_step)
                generated_observations_paths.append(paths)

            # get reward
            self.g_reward = self.get_rewards()

            for idx in range(self.envs.num_envs):
                done = self.current_dones[idx]

                self.num_steps_done += 1
                if done:
                    self.current_steps[idx] = 0

            if self.num_steps_done % 10 == 0:
                print(f"Exploration at {int(self.percent_done() * 100)}%")
        del self.current_dones
        del self.current_observations

        self.g_policy.to("cpu")
        self.g_rollouts.to("cpu")
        self.envs.close()
        return sorted(generated_observations_paths)

    def _training_log(
        self,
        writer,
        reward: float,
        losses: Dict[str, float],
        metrics: Dict[str, float] = None,
        prev_time: int = 0,
    ):
        writer.add_scalar(
            "reward",
            reward,
            self.num_steps_done,
        )

        writer.add_scalars(
            "losses",
            losses,
            self.num_steps_done,
        )

        if metrics:
            writer.add_scalars("metrics", metrics, self.num_steps_done)

    def train(self) -> None:
        self.envs = self._init_train()
        self.sub_goals = [[] for _ in range(self.envs.num_envs)]
        self.sub_goals_counter = [0] * self.envs.num_envs
        self.got_new_plan = [False] * self.envs.num_envs
        self.replan_retries = [0] * self.envs.num_envs
        self.replan = [True] * self.envs.num_envs
        self.astar_img = [None] * self.envs.num_envs

        self.first_step = True

        # goal policy observation space
        self.g_observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(len(BBSense.CLASSES), self.map_width, self.map_height),
            dtype='uint8',
        )

        # goal policy action space
        self.g_action_space = self.envs.action_spaces[0]

        # goal policy init
        device = torch.device("cuda:0" if self.config.ppo.cuda else "cpu")

        self.g_policy = RL_Policy(
            self.g_observation_space.shape,
            self.g_action_space,
            model_type=2,
            base_kwargs={
                'recurrent': self.config.ppo.use_recurrent_global,
                'hidden_size': self.config.ppo.g_hidden_size,
                'downscaling': self.config.ppo.global_downscaling,
            },
        ).to(device)

        # ppo agent init

        self.g_agent = PPO(
            self.g_policy,
            self.config.ppo.clip_param,
            self.config.ppo.ppo_epoch,
            self.envs.num_envs,
            self.config.ppo.value_loss_coeff,
            self.config.ppo.entropy_coef,
            lr=self.config.ppo.global_lr,
            eps=self.config.ppo.eps,
            max_grad_norm=self.config.ppo.max_grad_norm,
        )

        # rollout storage
        self.g_rollouts = GlobalRolloutStorage(
            self.config.ppo.num_global_steps,
            self.envs.num_envs,
            self.g_observation_space.shape,
            self.g_action_space,
            self.g_policy.rec_state_size,
            1,
        ).to(device)

        # create queues
        self.g_value_losses = deque(maxlen=1000)
        self.g_action_losses = deque(maxlen=1000)
        self.g_dist_entropies = deque(maxlen=1000)

        # first forward pass
        self.global_input = torch.rand(
            self.envs.num_envs, len(BBSense.CLASSES), self.map_height, self.map_width
        )
        self.g_rollouts.obs[0].copy_(self.global_input)
        self.g_masks = torch.ones(self.envs.num_envs).float().to(device)  # not used

        torch.set_grad_enabled(False)

        # run goal policy (predict global goals)
        self._step(self.envs)
        self.t_start = time.time()  # for tensorboard
        writer = TensorboardWriter(self.config.TENSORBOARD_DIR, flush_secs=30)

        self.checkpoint_folder = self.config.CHECKPOINT_FOLDER

        if not os.path.exists(self.checkpoint_folder):
            os.makedirs(self.checkpoint_folder)

        if self.config.ppo.load_checkpoint:
            if os.path.exists(self.config.ppo.load_checkpoint_path):
                self.g_policy.load_state_dict(
                    torch.load(self.config.ppo.load_checkpoint_path)
                )
            else:
                print("Error: checkpoint path does not exist.")
                return

        # training loop
        while not self.is_done():
            # step all envs
            self._step(self.envs)


            # get reward
            self.g_reward = self.get_rewards()

            # populate inputs
            # map, position input
            self.global_input = self.create_policy_inputs()

            # train goal policy
            if self.num_steps_done % self.config.ppo.num_global_steps == 0:

                torch.set_grad_enabled(True)

                self.g_next_value = self.g_policy.get_value(
                    self.g_rollouts.obs[-1],
                    self.g_rollouts.rec_states[-1],
                    self.g_rollouts.masks[-1],
                    extras=self.g_rollouts.extras[-1],
                ).detach()

                self.g_rollouts.compute_returns(
                    self.g_next_value,
                    self.config.ppo.use_gae,
                    self.config.ppo.gamma,
                    self.config.ppo.tau,
                )
                (
                    self.g_value_loss,
                    self.g_action_loss,
                    self.g_dist_entropy,
                ) = self.g_agent.update(self.g_rollouts)

                self.g_value_losses.append(self.g_value_loss)
                self.g_action_losses.append(self.g_action_loss)
                self.g_dist_entropies.append(self.g_dist_entropy)
                self.g_rollouts.after_update()

                torch.set_grad_enabled(False)

                losses = {
                    "value_loss": self.g_value_loss,
                    "action_loss": self.g_action_loss,
                }

                reward = self.g_reward.float().mean()

                self._training_log(writer, reward, losses)

            # save measures
            if (
                int(self.current_steps[0])
                % (self.config.ENVIRONMENT.MAX_EPISODE_STEPS - 2)
                == 0
            ):
                metrics = {
                    "episode_reward": self.current_infos[0]['episode_reward'],
                    "area_ratio": self.current_infos[0]['area_ratio'],
                }
                writer.add_scalars("metrics", metrics, self.num_steps_done)

            if self.num_steps_done % self.config.ppo.save_periodic == 0:
                torch.save(
                    self.g_policy.state_dict(),
                    os.path.join(
                        self.checkpoint_folder, f"checkpoint_{self.num_steps_done}.ckpt"
                    ),
                )

            self.num_steps_done += 1
            print("Steps done:", self.num_steps_done)

        self.envs.close()


class ObjectDetectorEnv:
    def __init__(self, config, device="cpu"):

        self.object_detector = MultiStageModel(**config)
        self.device = device

        self.object_detector.to(device)
        self.object_detector.eval()
        self.object_detector.model.roi_heads.box_predictor.box_predictor.test_score_thresh = (
            0.5
        )

    def load(self, load_path):
        if os.path.exists(load_path):
            self.object_detector.load_state_dict(torch.load(load_path)).to(self.device)

    def move_to_device(self, device='cpu'):
        self.device = device
        self.object_detector.to(device)

    def predict_batch(self, images):

        predictions = self.object_detector.infer(images)
        for i in range(len(images)):
            predictions[0][i]['instances'] = predictions[0][i]['instances'].to("cpu")

        return predictions
