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


class BCModelAsymmetric(nn.Module):
    """
    Asymmetric MLP: separate pathways for past vs future.
    Light noise on past to handle distribution shift.
    """

    def __init__(self, config: BCConfig) -> None:
        super().__init__()
        self.noise_std = 0.015

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


class BCModelTransformer(nn.Module):
    """
    Self-attention over future trajectory.
    Learns which upcoming targets matter most for current action.
    """

    def __init__(self, config: BCConfig) -> None:
        super().__init__()
        self.noise_std = 0.03

        # Future: embed each timestep, then self-attention
        self.future_embed = nn.Linear(4, 64)  # (target, roll, v_ego, a_ego) → 64

        # Positional encoding for future timesteps
        self.future_pos = nn.Parameter(torch.randn(1, 50, 64) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64,
            nhead=4,
            dim_feedforward=256,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.future_transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Past: simple MLP with heavy dropout
        self.past_encoder = nn.Sequential(
            nn.Linear(40, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.2),
        )

        # Current state
        self.current_encoder = nn.Sequential(
            nn.Linear(5, 32),
            nn.GELU(),
        )

        # Head: 64 (future) + 64 (past) + 32 (current) + 2 (timestep) = 162
        self.head = nn.Sequential(
            nn.Linear(162, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        current = x[:, 0:5]
        past = x[:, 5:45]
        future = x[:, 45:245].reshape(-1, 50, 4)  # [B, 50, 4]
        timestep = x[:, 245:247]

        # Noise on past
        if self.training:
            past = past + torch.randn_like(past) * self.noise_std

        # Future processing with attention
        future_emb = self.future_embed(future) + self.future_pos  # [B, 50, 64]
        future_out = self.future_transformer(future_emb)  # [B, 50, 64]
        future_pooled = future_out.mean(dim=1)  # [B, 64]

        past_features = self.past_encoder(past)
        current_features = self.current_encoder(current)

        combined = torch.cat(
            [future_pooled, past_features, current_features, timestep], dim=1
        )
        return torch.tanh(self.head(combined)).squeeze(-1) * 2.0

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class BCModelBig(nn.Module):
    """
    Large capacity, heavy regularization everywhere.
    """

    def __init__(self, config: BCConfig) -> None:
        super().__init__()
        self.noise_std = 0.02
        self.past_drop_prob = 0.1  # Sometimes zero out entire past

        self.net = nn.Sequential(
            nn.BatchNorm1d(247),
            nn.Linear(247, 2048),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 2048),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            # Noise on past
            noise = torch.randn_like(x[:, 5:45]) * self.noise_std
            x = x.clone()
            x[:, 5:45] = x[:, 5:45] + noise

            # Randomly zero out entire past history (forces reliance on future)
            mask = torch.rand(x.shape[0], 1, device=x.device) > self.past_drop_prob
            x[:, 5:45] = x[:, 5:45] * mask

        return torch.tanh(self.net(x).squeeze(-1)) * 2.0

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
