import numpy as np
import torch
import torch.nn as nn


class BasicCNN(nn.Module):
    def __init__(self, channel_dim: int = 4, state_dim: int = 64, init_weights=True):
        super(BasicCNN, self).__init__()

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
        self.feature_dim = 16 * (state_dim // 8) ** 2

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.flatten(x, start_dim=1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)


class BilinearCNN(nn.Module):
    def __init__(self, channel_dim: int = 4, init_weights=True):
        super(BilinearCNN, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(channel_dim, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ELU(),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.feature_dim = 1024

        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size()[0]
        x = self.conv(x)

        feature_map_size = x.size()[-1] ** 2
        x = torch.flatten(x, start_dim=2)
        feature_map_count = x.size()[1]
        x = torch.bmm(x, torch.transpose(x, 1, 2)) / feature_map_size  # Bilinear
        x = x.view(batch_size, feature_map_count**2)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, np.sqrt(2))
                nn.init.constant_(m.bias, 0.0)
