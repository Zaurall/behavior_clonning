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
        out = self.net(x).squeeze(-1)
        return torch.tanh(out) * 2.0

    def count_parameters(self) -> int:
        """Count trainable params"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BCModelCNN(nn.Module):
    def __init__(self, config: BCConfig) -> None:
        super().__init__()

        # Conv over past (20 steps × 2 channels: actions, lataccels)
        self.past_conv = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Flatten(),  # 32 × 20 = 640
        )

        # Conv over future (50 steps × 4 channels: target, roll, v_ego, a_ego)
        self.future_conv = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Flatten(),  # 32 × 50 = 1600
        )

        # current(5) + past(640) + future(1600) + timestep(2) = 2247
        self.head = nn.Sequential(
            nn.Linear(2247, 512),
            nn.GELU(),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current = x[:, 0:5]
        past_actions = x[:, 5:25]
        past_lataccels = x[:, 25:45]
        future = x[:, 45:245].reshape(-1, 4, 50)  # 4 channels × 50 steps
        timestep = x[:, 245:247]

        past = torch.stack([past_actions, past_lataccels], dim=1)  # [B, 2, 20]

        past_features = self.past_conv(past)
        future_features = self.future_conv(future)

        combined = torch.cat([current, past_features, future_features, timestep], dim=1)
        return torch.tanh(self.head(combined)).squeeze(-1) * 2.0

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
