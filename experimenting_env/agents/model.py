import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

from experimenting_env.sensor_data import BBSense
from experimenting_env.utils.distributions import Categorical, DiagGaussian
from experimenting_env.utils.model import (
    ChannelPool,
    Flatten,
    NNBase,
    get_grid,
)


# Global Policy model code
class Global_Policy(NNBase):
    def __init__(
        self,
        input_shape,
        recurrent=False,
        hidden_size=512,
        downscaling=1,
        input_channels=2,
    ):
        super(Global_Policy, self).__init__(recurrent, hidden_size, hidden_size)

        out_size = int(input_shape[1] / 16.0 * input_shape[2] / 16.0)

        self.main = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(input_channels, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            Flatten(),
        )
        self.linear1 = nn.Linear(out_size * 32 + 8, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 256)
        self.critic_linear = nn.Linear(256, 1)
        self.orientation_emb = nn.Embedding(72, 8)

        with torch.no_grad():
            self.linear1.weight.data.uniform_(-0.003, 0.003)
            self.linear2.weight.data.uniform_(-0.003, 0.003)
            self.critic_linear.weight.data.uniform_(-0.003, 0.003)

        self.train()

    def forward(self, inputs, rnn_hxs, masks, extras):
        x = self.main(inputs)
        # print("MAIN EXTRA", x.shape, self.orientation_emb(extras).squeeze(1).shape)
        orientation_emb = self.orientation_emb(extras).squeeze(1)
        x = torch.cat((x, orientation_emb), 1)

        x = nn.ReLU()(self.linear1(x))
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        x = nn.ReLU()(self.linear2(x))

        return self.critic_linear(x).squeeze(-1), x, rnn_hxs


# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/model.py#L15
class RL_Policy(nn.Module):
    def __init__(
        self,
        obs_shape,
        action_space,
        model_type=0,  # 0 = 2 inputs, 1 = 8 inputs
        base_kwargs=None,
    ):

        super(RL_Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if model_type == 0:
            input_channels = 2

        elif model_type == 1:
            input_channels = 8
        elif model_type == 2:
            input_channels = len(BBSense.CLASSES)  # n-classes
        else:
            raise NotImplementedError

        self.network = Global_Policy(
            obs_shape, input_channels=input_channels, **base_kwargs
        )

        if action_space.__class__.__name__ == "Discrete" or action_space.__class__.__name__ == "ActionSpace":
            num_outputs = action_space.n
            self.dist = Categorical(self.network.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.network.output_size, num_outputs)
        else:
            raise NotImplementedError

        self.model_type = model_type

    @property
    def is_recurrent(self):
        return self.network.is_recurrent

    @property
    def rec_state_size(self):
        """Size of rnn_hx."""
        return self.network.rec_state_size

    def forward(self, inputs, rnn_hxs, masks, extras):
        if extras is None:
            return self.network(inputs, rnn_hxs, masks)
        else:
            return self.network(inputs, rnn_hxs, masks, extras)

    def act(self, inputs, rnn_hxs, masks, extras=None, deterministic=False):
        value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks, extras)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks, extras=None):
        value, _, _ = self(inputs, rnn_hxs, masks, extras)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action, extras=None):

        value, actor_features, rnn_hxs = self(inputs, rnn_hxs, masks, extras)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs
