from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch

from offline.config import PGTOConfig


@dataclass
class FutureContext:
    targets: torch.Tensor  # [20]
    roll: torch.Tensor  # [20]
    v_ego: torch.Tensor  # [20]
    a_ego: torch.Tensor  # [20]


@dataclass
class Segment:
    """Preprocessed segment data ready for PGTO optimization."""

    segment_id: str

    # Target trajectory [T]
    targets: torch.Tensor

    # Vehicle state trajectories [T]
    roll: torch.Tensor
    v_ego: torch.Tensor
    a_ego: torch.Tensor

    # Initial conditions from warmup period
    initial_history_states: torch.Tensor  # [20, 4]
    initial_history_lataccel: torch.Tensor  # [20]
    initial_lataccel: float

    num_steps: int  # T

    def get_future_context(self, t: int, horizon: int) -> FutureContext:
        """
        Get future context for timestamp t with horizon
        """

        def pad_to_horizon(tensor: torch.Tensor) -> torch.Tensor:
            future = tensor[t : t + horizon]
            if len(future) < horizon:
                pad_len = horizon - len(future)
                future = torch.cat([future, future[-1:].expand(pad_len)])

            return future

        return FutureContext(
            targets=pad_to_horizon(self.targets),
            roll=pad_to_horizon(self.roll),
            v_ego=pad_to_horizon(self.v_ego),
            a_ego=pad_to_horizon(self.a_ego),
        )


def load_segment(segment_path: Path, config: PGTOConfig) -> Segment:
    """
    Load and preprocess a segment from its CSV.
    """

    df = pd.read_csv(segment_path)

    # Control period starts after warmup
    control_start = config.control_start_idx
    warmup_start = control_start - config.context_length

    # Target trajectory
    targets = torch.tensor(
        df["targetLateralAcceleration"].values[control_start:],
        dtype=torch.float32,
        device=config.device,
    )

    # Vehicle states
    roll = torch.tensor(
        df["roll"].values[control_start:],
        dtype=torch.float32,
        device=config.device,
    )
    v_ego = torch.tensor(
        df["vEgo"].values[control_start:],
        dtype=torch.float32,
        device=config.device,
    )
    a_ego = torch.tensor(
        df["aEgo"].values[control_start:],
        dtype=torch.float32,
        device=config.device,
    )

    # Warmup actions
    warmup_actions = df["steerCommand"].values[warmup_start:control_start]

    # Warmup states
    warmup_roll = df["roll"].values[warmup_start:control_start]
    warmup_v_ego = df["vEgo"].values[warmup_start:control_start]
    warmup_a_ego = df["aEgo"].values[warmup_start:control_start]

    initial_history_states = initial_history_states = torch.stack(
        [
            torch.tensor(warmup_actions, dtype=torch.float32, device=config.device),
            torch.tensor(warmup_roll, dtype=torch.float32, device=config.device),
            torch.tensor(warmup_v_ego, dtype=torch.float32, device=config.device),
            torch.tensor(warmup_a_ego, dtype=torch.float32, device=config.device),
        ],
        dim=1,
    )

    warmup_lataccel = df["targetLateralAcceleration"].values[warmup_start:control_start]
    initial_history_lataccel = torch.tensor(
        warmup_lataccel,
        dtype=torch.float32,
        device=config.device,
    )

    initial_lataccel = float(warmup_lataccel[-1])

    return Segment(
        segment_id=segment_path.stem,
        targets=targets,
        roll=roll,
        v_ego=v_ego,
        a_ego=a_ego,
        initial_history_states=initial_history_states,
        initial_history_lataccel=initial_history_lataccel,
        initial_lataccel=initial_lataccel,
        num_steps=len(targets),
    )


def get_segment_paths(segments_dir: Path) -> list[Path]:
    """Get all segment CSV paths sorted by name."""
    return sorted(segments_dir.glob("*.csv"))
