from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np
import torch
from tqdm import tqdm

from offline.cmaes import CMAESModel, CMAESState
from offline.config import PGTOConfig
from offline.pgto.step import PGTOStep
from offline.physics import BatchedPhysics
from offline.segment import Segment


@dataclass
class TrajectoryData:
    """
    Full trajectory data for one restart.
    """

    # Observations at each timestamp
    history_states: torch.Tensor  # [T, 20, 4]
    history_tokens: torch.Tensor  # [T, 20]
    current_lataccel: torch.Tensor  # [T]
    targets: torch.Tensor  # [T]
    cmaes_state: torch.Tensor  # [T, 4] - (prev_error, error_integral, u_t1, u_t2)

    # Actions taken
    actions: torch.Tensor  # [T]

    def to_dict(self) -> dict[str, torch.Tensor]:
        return {
            "history_states": self.history_states.cpu(),
            "history_tokens": self.history_tokens.cpu(),
            "current_lataccel": self.current_lataccel.cpu(),
            "targets": self.targets.cpu(),
            "cmaes_state": self.cmaes_state.cpu(),
            "actions": self.actions.cpu(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, torch.Tensor], device: str = "cpu") -> Self:
        return cls(
            history_states=d["history_states"].to(device),
            history_tokens=d["history_tokens"].to(device),
            current_lataccel=d["current_lataccel"].to(device),
            targets=d["targets"].to(device),
            cmaes_state=d["cmaes_state"].to(device),
            actions=d["actions"].to(device),
        )


@dataclass
class RestartResult:
    """
    Result from one restart.
    """

    trajectory: TrajectoryData
    cost: float


@dataclass
class PGTOResult:
    """
    Complete result from PGTO optimization.
    """

    segment_id: str
    restarts: list[RestartResult]
    best_restart_idx: int
    best_cost: float

    def save(self, path: Path) -> None:
        data = {
            "segment_id": self.segment_id,
            "num_restarts": len(self.restarts),
            "best_restart_idx": self.best_restart_idx,
            "best_cost": self.best_cost,
        }

        for i, restart in enumerate(self.restarts):
            data[f"restart_{i}_trajectory"] = restart.trajectory.to_dict()
            data[f"restart_{i}_cost"] = restart.cost

        torch.save(data, path)

    @classmethod
    def load(cls, path: Path, device: str = "cpu") -> Self:
        data = torch.load(path, map_location=device)

        restarts = []
        for i in range(data["num_restarts"]):
            trajectory = TrajectoryData.from_dict(
                data[f"restart_{i}_trajectory"], device
            )
            restarts.append(
                RestartResult(
                    trajectory=trajectory,
                    cost=data[f"restart_{i}_cost"],
                )
            )

        return cls(
            segment_id=data["segment_id"],
            restarts=restarts,
            best_restart_idx=data["best_restart_idx"],
            best_cost=data["best_cost"],
        )


class PGTOOptimizer:
    """
    Policy-Guided Trajectory Optimization.

    Runs R parallel restarts through the segment, each finding optimal actions
    via stochastic noise around a base CMA-ES model.
    """

    def __init__(self, config: PGTOConfig) -> None:
        self.config = config

        self.physics = BatchedPhysics(config=config)
        self.cmaes = CMAESModel(config=config)

        # PGTO step optimizer
        self.pgto_step = PGTOStep(
            physics=self.physics,
            cmaes=self.cmaes,
            config=config,
        )

    def optimize(self, segment: Segment, verbose: bool = False) -> PGTOResult:
        """
        Run PGTO optimization on a segment.

        Args:
            segment: Preprocessed segment data
            verbose: Show progress bar

        Returns:
            PGTOResult with all restart trajectories and costs
        """
        trajectories, costs = self._run_parallel_passes(segment, verbose)
        results = [
            RestartResult(trajectory=traj, cost=cost.item())
            for traj, cost in zip(trajectories, costs)
        ]

        costs = [r.cost for r in results]
        best_idx = int(np.argmin(costs))

        return PGTOResult(
            segment_id=segment.segment_id,
            restarts=results,
            best_restart_idx=best_idx,
            best_cost=costs[best_idx],
        )

    def _run_parallel_passes(
        self, segment: Segment, verbose: bool
    ) -> tuple[list[TrajectoryData], torch.Tensor]:
        """
        Run R parallel restarts through the segment.

         At each timestep:
            1. Record current observations for all R restarts
            2. Run a step of PGTO to find best action for each restart
            3. Advance all R trajectories with stochastic physics
        """

        R = self.config.num_restarts
        T = segment.num_steps
        device = self.config.device

        # Init R parallel states (all start the same)
        history_states = (
            segment.initial_history_states.unsqueeze(0).expand(R, -1, -1).clone()
        )
        history_tokens = (
            self.physics.tokenize(segment.initial_history_lataccel)
            .unsqueeze(0)
            .expand(R, -1)
            .clone()
        )
        prev_lataccel = torch.full((R,), segment.initial_lataccel, device=device)

        init_u_t1 = segment.initial_history_states[-1, 0].expand(R)
        init_u_t2 = segment.initial_history_states[-2, 0].expand(R)

        cmaes_state = CMAESState(
            prev_error=torch.zeros(R, device=device),
            error_integral=torch.zeros(R, device=device),
            u_t1=init_u_t1,
            u_t2=init_u_t2,
        )

        # Storage for trajectories
        all_history_states = torch.zeros(R, T, 20, 4, device=device)
        all_history_tokens = torch.zeros(R, T, 20, dtype=torch.long, device=device)
        all_current_lataccel = torch.zeros(R, T, device=device)
        all_targets = torch.zeros(R, T, device=device)
        all_cmaes_state = torch.zeros(R, T, 4, device=device)
        all_actions = torch.zeros(R, T, device=device)

        prev_action = history_states[:, -1, 0].clone()

        # Accumulators
        total_tracking = torch.zeros(R, device=device)
        total_jerk = torch.zeros(R, device=device)

        # Sequential pass
        iterator = range(T)
        if verbose:
            iterator = tqdm(iterator, desc="PGTO", leave=False)

        for t in iterator:
            target_t = segment.targets[t]
            future_context = segment.get_future_context(t, self.config.horizon)

            # Record current state BEFORE taking action
            all_history_states[:, t] = history_states
            all_history_tokens[:, t] = history_tokens
            all_current_lataccel[:, t] = prev_lataccel
            all_targets[:, t] = target_t
            all_cmaes_state[:, t, 0] = cmaes_state.prev_error
            all_cmaes_state[:, t, 1] = cmaes_state.error_integral
            all_cmaes_state[:, t, 2] = cmaes_state.u_t1
            all_cmaes_state[:, t, 3] = cmaes_state.u_t2

            error_t = target_t - prev_lataccel
            cmaes_state.error_integral = torch.clamp(
                cmaes_state.error_integral + error_t, -5.0, 5.0
            )

            # Step PGTO, find best action for each restart
            # Deterministic physics inside
            # Lots of comments on physics because it's hard to keep track of 😂
            best_actions = self.pgto_step.optimize(
                history_states=history_states,
                history_tokens=history_tokens,
                prev_lataccel=prev_lataccel,
                prev_action=prev_action,
                cmaes_state=cmaes_state,
                future_context=future_context,
            )

            all_actions[:, t] = best_actions

            # Advance trajectories with stochastic physics
            new_state = torch.stack(
                [
                    best_actions,
                    segment.roll[t].expand(R),
                    segment.v_ego[t].expand(R),
                    segment.a_ego[t].expand(R),
                ],
                dim=-1,
            )
            history_states = torch.cat(
                [history_states[:, 1:, :], new_state.unsqueeze(1)], dim=1
            )

            new_lataccel = self.physics.sample(
                history_states, history_tokens, prev_lataccel
            )

            # Accumulate cost
            if t < self.config.cost_steps:
                tracking_error = new_lataccel - segment.targets[t]
                jerk = (new_lataccel - prev_lataccel) / 0.1
                total_tracking += tracking_error**2
                total_jerk += jerk**2

            # Update CMA-ES state
            error_t = target_t - prev_lataccel
            cmaes_state.prev_error = error_t
            cmaes_state.u_t2 = cmaes_state.u_t1.clone()
            cmaes_state.u_t1 = best_actions

            # Update tokens
            new_tokens = self.physics.tokenize(new_lataccel)
            history_tokens = torch.cat(
                [history_tokens[:, 1:], new_tokens.unsqueeze(1)], dim=1
            )

            prev_lataccel = new_lataccel
            prev_action = best_actions

        costs = self.config.w_tracking * (
            total_tracking / self.config.cost_steps
        ) + self.config.w_jerk * (total_jerk / self.config.cost_steps)

        # Build into TrajectoryData
        trajectories = []
        for r in range(R):
            trajectories.append(
                TrajectoryData(
                    history_states=all_history_states[r],
                    history_tokens=all_history_tokens[r],
                    current_lataccel=all_current_lataccel[r],
                    targets=all_targets[r],
                    cmaes_state=all_cmaes_state[r],
                    actions=all_actions[r],
                )
            )

        return trajectories, costs
