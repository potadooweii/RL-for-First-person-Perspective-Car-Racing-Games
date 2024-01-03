import numpy as np
import torch
import torch.nn as nn
from models.backbone import BasicCNN, BilinearCNN
from typing import Optional


class ActorNet(nn.Module):
    def __init__(
        self,
        channel_dim: int = 4,
        state_dim: int = 64,
        action_dim: int = 2,
        backbone: Optional[str] = None,
        motor_magnitude: float = 0.05,
    ):
        super(ActorNet, self).__init__()

        # Init NN
        self.conv = (
            BilinearCNN(channel_dim)
            if backbone == "bilinear"
            else BasicCNN(channel_dim, state_dim)
        )
        self.feature_dim = self.conv.feature_dim
        self.motor_magnitude = motor_magnitude

        self.linear = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.LayerNorm(256),
            nn.ELU(),
            nn.Linear(256, action_dim),
            nn.LayerNorm(action_dim),
            nn.Tanh(),
        )

    def forward(self, state) -> torch.Tensor:
        state = state.float() / 255.0
        h = self.conv(state)
        h = self.linear(h)

        h_clone: torch.Tensor = h.clone()
        # map to valid action space
        h_clone[:, 0] = ((h_clone[:, 0] + 1) / 2) * self.motor_magnitude

        return h_clone


class CriticNet(nn.Module):
    def __init__(
        self,
        channel_dim: int = 4,
        state_dim: int = 64,
        action_dim: int = 2,
        backbone: Optional[str] = None,
    ) -> None:
        super(CriticNet, self).__init__()

        # Init NN
        self.conv = (
            BilinearCNN(channel_dim)
            if backbone == "bilinear"
            else BasicCNN(channel_dim, state_dim)
        )
        self.feature_dim = self.conv.feature_dim

        self.action_linear = nn.Sequential(
            nn.Linear(action_dim, 256), nn.LayerNorm(256), nn.ELU()
        )

        self.state_linear = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.LayerNorm(256),
            nn.ELU(),
        )

        self.concat_linear = nn.Sequential(
            nn.Linear(512, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Linear(64, 64),
            nn.LayerNorm(64),
            nn.ELU(),
            nn.Linear(64, 1),
        )

    def forward(self, state, action) -> torch.Tensor:
        # extract the state features
        state = state.float() / 255.0
        state_h = self.conv(state)

        # state features
        state_h = self.state_linear(state_h)
        # action features
        action_h = self.action_linear(action)

        # concat
        h = self.concat_linear(torch.concat((state_h, action_h), dim=1))

        return h
