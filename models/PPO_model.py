import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, normal
from models.backbone import BasicCNN, BilinearCNN
from typing import Optional, List


class PPONet(nn.Module):
    def __init__(
        self,
        channel_dim: int,
        state_dim: int,
        action_dim: int,
        backbone: Optional[str] = None,
        continuous: bool = False,
        init_weights=True,
        motor_magnitude: float = 0.05,
    ):
        super(PPONet, self).__init__()

        # Init global variables
        self.continuous = continuous
        self.motor_magnitude = motor_magnitude

        # Init NN
        # self.conv = (
        #     BilinearCNN(channel_dim)
        #     if backbone == "bilinear"
        #     else BasicCNN(channel_dim, state_dim)
        # )
        # self.feature_dim = self.conv.feature_dim
        
        self.conv = nn.Sequential(
            nn.Conv2d(channel_dim, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
        )

        # actor
        self.actor = nn.Sequential(
            nn.Linear(16 * (state_dim // 8) ** 2, 256),
            # nn.Linear(self.feature_dim, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, action_dim),
            nn.LayerNorm(action_dim),
            nn.Tanh(),
        )
        # if self.continuous:
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

        # critic
        self.critic = nn.Sequential(
            nn.Linear(16 * (state_dim // 8) ** 2, 256),
            # nn.Linear(self.feature_dim, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, 1),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = x.float() / 255.0
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)

        # get value
        value = self.critic(x)
        value = torch.squeeze(value)

        # get action distribution
        logits: torch.Tensor = self.actor(x)

        return logits

    def get_ppo_output(self, x, eval=False, old_action: Optional[List] = None):
        x = x.float() / 255.0
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)

        # get value
        value = self.critic(x)
        value = torch.squeeze(value)

        # get action distribution
        logits: torch.Tensor = self.actor(x)
        if self.continuous:
            # map to valid action space
            mu: torch.Tensor = logits.clone()
            mu[:, 0] = ((mu[:, 0] + 1) / 2) * self.motor_magnitude
            std = self.log_std.exp()
            dist = normal.Normal(mu, std)
        else:
            dist = Categorical(logits=logits)

        # get action
        action = (
            torch.squeeze(mu)
            if self.continuous
            else torch.argmax(logits)
            if eval
            else dist.sample()
        )

        # get action log probability
        if self.continuous:
            action_logprob = (
                dist.log_prob(action).sum(1)
                if old_action is None
                else dist.log_prob(old_action).sum(1)
            )
            entropy = dist.entropy().sum(1)
        else:
            action_logprob = (
                dist.log_prob(action) if old_action is None else dist.log_prob(old_action)
            )
            entropy = dist.entropy()

        return action, action_logprob, value, entropy

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
