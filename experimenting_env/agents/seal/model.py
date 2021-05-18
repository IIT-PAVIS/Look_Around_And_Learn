import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn import functional as F

from experimenting_env.agents.model import Global_Policy
from experimenting_env.utils.distributions import Categorical, DiagGaussian
from experimenting_env.utils.model import (
    ChannelPool,
    Flatten,
    NNBase,
    get_grid,
)

from .pointnet import PCDFeat


# Global Policy model code
class Global_Policy_PCD(NNBase):
    def __init__(self, input_shape, recurrent=False, hidden_size=512, downscaling=1):
        super().__init__(recurrent, hidden_size, hidden_size)

        out_size = 128

        self.main = PCDFeat()

        self.linear1 = nn.Linear(out_size + 8, hidden_size)
        self.linear2 = nn.Linear(hidden_size, 128)
        self.critic_linear = nn.Linear(128, 1)
        self.orientation_emb = nn.Embedding(72, 8)

        with torch.no_grad():
            self.linear1.weight.data.uniform_(-0.003, 0.003)
            self.linear2.weight.data.uniform_(-0.003, 0.003)
            self.critic_linear.weight.data.uniform_(-0.003, 0.003)

        self.train()

    def forward(self, inputs, rnn_hxs, masks, extras):

        device = inputs.device

        x = self.main(inputs)

        orientation_emb = self.orientation_emb(extras).squeeze(1)
        x = torch.cat((x, orientation_emb), 1)

        x = nn.ReLU()(self.linear1(x))
        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        x = nn.ReLU()(self.linear2(x))

        return self.critic_linear(x).squeeze(-1), x, rnn_hxs


class RL_Policy_PCD(nn.Module):
    def __init__(
        self,
        obs_shape,
        action_space,
        model_type=0,  # 0 = global, action_space = Box
        base_kwargs=None,
    ):

        super().__init__()
        if base_kwargs is None:
            base_kwargs = {}

        if model_type == 0:
            self.network = Global_Policy_PCD(obs_shape, **base_kwargs)
        else:
            raise NotImplementedError

        if action_space.__class__.__name__ == "Discrete":
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
        try:
            dist = self.dist(actor_features)
        except:
            dist = self.dist(actor_features + 0.01)

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
