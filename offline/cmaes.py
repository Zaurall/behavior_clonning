from dataclasses import dataclass
# from typing import Self

import numpy as np
import torch
import torch.nn as nn

from offline.config import PGTOConfig

INPUT_SIZE = 14
HIDDEN_1 = 20
HIDDEN_2 = 16


@dataclass
class CMAESState:
    """
    CMA-ES controller state during rollout.
    """

    prev_error: torch.Tensor  # [B]
    error_integral: torch.Tensor  # [B]
    u_t1: torch.Tensor  # [B]
    u_t2: torch.Tensor  # [B]

    @classmethod
    # def zeros(cls, batch_size: int, device: str) -> Self:
    def zeros(cls, batch_size: int, device: str) -> "CMAESState":
        return cls(
            prev_error=torch.zeros(batch_size, device=device),
            error_integral=torch.zeros(batch_size, device=device),
            u_t1=torch.zeros(batch_size, device=device),
            u_t2=torch.zeros(batch_size, device=device),
        )

    @classmethod
    def from_single(
        cls,
        prev_error: float,
        error_integral: float,
        u_t1: float,
        u_t2: float,
        batch_size: int,
        device: str,
    ) -> "CMAESState":
    # ) -> Self:
        return cls(
            prev_error=torch.full((batch_size,), prev_error, device=device),
            error_integral=torch.full((batch_size,), error_integral, device=device),
            u_t1=torch.full((batch_size,), u_t1, device=device),
            u_t2=torch.full((batch_size,), u_t2, device=device),
        )

    # def clone(self) -> Self:
    def clone(self) -> "CMAESState":
        return type(self)(
            prev_error=self.prev_error.clone(),
            error_integral=self.error_integral.clone(),
            u_t1=self.u_t1.clone(),
            u_t2=self.u_t2.clone(),
        )

    # def expand(self, new_batch_size: int) -> Self:
    def expand(self, new_batch_size: int) -> "CMAESState":
        """
        Expand batch dimension (for going from R restarts to R*K candidates).
        """
        return type(self)(
            prev_error=self.prev_error.repeat_interleave(
                new_batch_size // self.prev_error.shape[0]
            ),
            error_integral=self.error_integral.repeat_interleave(
                new_batch_size // self.error_integral.shape[0]
            ),
            u_t1=self.u_t1.repeat_interleave(new_batch_size // self.u_t1.shape[0]),
            u_t2=self.u_t2.repeat_interleave(new_batch_size // self.u_t2.shape[0]),
        )


class CMAESModel(nn.Module):
    """
    Pretrained CMA-ES controller
    """

    def __init__(self, config: PGTOConfig) -> None:
        super().__init__()

        self.config = config

        self.device = self.config.device

        self.net = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_1),
            nn.Tanh(),
            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.Tanh(),
            nn.Linear(HIDDEN_2, 1),
            nn.Tanh(),
        )

        self._load_params()
        self.to(self.device)
        self.eval()

    def _load_params(self) -> None:
        """
        Load params from numpy file.
        """
        params = np.load(self.config.cmaes_params_path)
        idx = 0

        for layer in [self.net[0], self.net[2], self.net[4]]:
            assert isinstance(layer, nn.Linear)  # Satisfies type checker

            in_f, out_f = layer.in_features, layer.out_features

            # Training used x @ W, PyTorch uses x @ W.T, so transpose
            w = params[idx : idx + in_f * out_f].reshape(in_f, out_f)
            layer.weight.data = torch.tensor(w.T, dtype=torch.float32)
            idx += in_f * out_f

            b = params[idx : idx + out_f]
            layer.bias.data = torch.tensor(b, dtype=torch.float32)
            idx += out_f

    @torch.no_grad()
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Predict steering action from features.

        Args:
            features: [B, 14]

        Returns:
            actions: [B] in range [-2, 2]
        """
        return self.net(features).squeeze(-1) * 2.0

    def compute_features(
        self,
        target: torch.Tensor,  # [B]
        current_lataccel: torch.Tensor,  # [B]
        state: CMAESState,
        v_ego: torch.Tensor,  # [B]
        a_ego: torch.Tensor,  # [B]
        roll: torch.Tensor,  # [B]
        future_targets: torch.Tensor,  # [H] - shared across batch
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute features for CMA-ES model.

        Args:
            target: Current target lataccel [B]
            current_lataccel: Current actual lataccel [B]
            state: CMA-ES state containing prev_error, error_integral, u_t1, u_t2
            v_ego: Vehicle velocity [B]
            a_ego: Vehicle acceleration [B]
            roll: Road roll [B]
            future_targets: Future target trajectory [H], shared across batch

        Returns:
            features: [B, 14]
        """
        # PID-like terms
        error = target - current_lataccel
        error_deriv = error - state.prev_error

        # Future statistics (shared across batch)
        n = len(future_targets)
        target_mean = target.mean()

        future_near = future_targets[:5].mean() if n >= 1 else target_mean
        future_mid = future_targets[5:15].mean() if n > 5 else future_near
        future_far = future_targets[15:40].mean() if n > 15 else future_mid
        future_var = (
            future_targets.std(unbiased=False)
            if n > 1
            else torch.tensor(0.0, device=self.device)
        )
        future_max_abs = future_targets.abs().max() if n >= 1 else target.abs().max()

        # Stack features [B, 14]
        features = torch.stack(
            [
                error / 5.0,
                error_deriv / 2.0,
                state.error_integral / 5.0,
                target / 5.0,
                v_ego / 30.0,
                a_ego / 4.0,
                roll / 2.0,
                state.u_t1 / 2.0,
                state.u_t2 / 2.0,
                (future_near / 5).expand_as(error),
                (future_mid / 5).expand_as(error),
                (future_far / 5).expand_as(error),
                (future_var / 5).expand_as(error),
                (future_max_abs / 5).expand_as(error),
            ],
            dim=-1,
        )

        return features
