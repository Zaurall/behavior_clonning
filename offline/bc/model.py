import torch
import torch.nn as nn

from offline.config import BCConfig


class BCModel(nn.Module):
    """
    Asymmetric MLP with separate pathways for past vs future.
    Light noise on past to handle distribution shift.
    """

    def __init__(self, config: BCConfig) -> None:
        super().__init__()

        self.config = config

        self.noise_std = self.config.past_noise_std

        # Past pathway: keep relatively small (distribution shift at eval)
        self.past_encoder = nn.Sequential(
            nn.Linear(40, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
        )

        # Future pathway: large (no distribution shift, reliable signal)
        self.future_encoder = nn.Sequential(
            nn.Linear(200, 1024),
            nn.GELU(),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
        )

        # Current state: small
        self.current_encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.GELU(),
            nn.Linear(64, 64),
            nn.GELU(),
        )

        # Head: 128 + 256 + 64 + 2 (past validity) = 450
        self.head = nn.Sequential(
            nn.Linear(450, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current = x[:, 0:5]
        past = x[:, 5:45]
        future = x[:, 45:245]
        timestep = x[:, 245:247]

        if self.training and self.noise_std > 0:
            past = past + torch.randn_like(past) * self.noise_std

        past_features = self.past_encoder(past)
        future_features = self.future_encoder(future)
        current_features = self.current_encoder(current)

        combined = torch.cat(
            [past_features, future_features, current_features, timestep], dim=1
        )
        return torch.tanh(self.head(combined)).squeeze(-1) * 2.0

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
