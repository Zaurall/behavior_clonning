import torch
import torch.nn as nn

from offline.config import BCConfig


class BCModel(nn.Module):
    """
    MLP for BC.

    LayerNorm -> (Linear + GELU) x N -> Linear -> Tanh x 2.0
    """

    def __init__(self, config: BCConfig) -> None:
        super().__init__()

        self.config = config

        layers: list[nn.Module] = []

        # Input norm
        layers.append(nn.BatchNorm1d(config.input_size))

        # Hidden layers
        prev_size = config.input_size
        for hidden_size in config.hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.GELU())
            prev_size = hidden_size

        # Output layer
        layers.append(nn.Linear(prev_size, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: [batch, 247] input features

        Returns:
            actions: [batch] in range [-2, 2]
        """

        if self.training and self.config.past_noise_std > 0:
            noise = torch.randn_like(x[:, 5:45]) * self.config.past_noise_std
            x = x.clone()
            x[:, 5:45] = x[:, 5:45] + noise

        out = self.net(x).squeeze(-1)

        return torch.tanh(out) * 2.0

    def count_parameters(self) -> int:
        """Count trainable params"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BCModelAsymmetric(nn.Module):
    """
    Asymmetric MLP: separate pathways for past vs future.
    Light noise on past to handle distribution shift.
    """

    def __init__(self, config: BCConfig) -> None:
        super().__init__()
        self.noise_std = 0.025

        # Past pathway: 40 inputs (20 actions + 20 lataccels)
        self.past_encoder = nn.Sequential(
            nn.Linear(40, 256),
            nn.GELU(),
            nn.Linear(256, 128),
            nn.GELU(),
        )

        # Future pathway: 200 inputs (50 × 4)
        self.future_encoder = nn.Sequential(
            nn.Linear(200, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
        )

        # Current state: 5 inputs
        self.current_encoder = nn.Sequential(
            nn.Linear(5, 64),
            nn.GELU(),
        )

        # Head: 128 + 256 + 64 + 2 = 450
        self.head = nn.Sequential(
            nn.Linear(450, 256),
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
