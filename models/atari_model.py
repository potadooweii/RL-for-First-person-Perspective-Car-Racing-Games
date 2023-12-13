import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, normal

class AtariNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, init_weights=True):
        super(AtariNet, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
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
            nn.Linear(16*(state_dim//8)**2, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, action_dim),
            nn.LayerNorm(action_dim),
            nn.Tanh()
        )
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))

        # critic
        self.critic = nn.Sequential(
            nn.Linear(16*(state_dim//8)**2, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, 1),
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x, eval=False, old_action=[]):
        x = x.float() / 255.
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        value = self.critic(x)
        value = torch.squeeze(value)

        mu = self.actor(x)
        mu_clone = mu.clone()
        # map to valid action space
        mu_clone[:, 0] = (mu_clone[:, 0]+1)*0.5 + 0.1
        std = self.log_std.exp()
        dist = normal.Normal(mu, std)

        ### TODO ###
        # Finish the forward function
        # Return action, action probability, value, entropy

        if eval:
            action = torch.squeeze(mu)
        else:
            action = dist.sample()

        if len(old_action) == 0:
            action_logprob = dist.log_prob(action).sum(1)
        else:
            action_logprob = dist.log_prob(old_action).sum(1)
        
        entropy = dist.entropy().sum(1)

        return action, action_logprob, value, entropy

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
                


