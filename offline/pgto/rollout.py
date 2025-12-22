import torch

from offline.cmaes import CMAESModel, CMAESState
from offline.config import PGTOConfig
from offline.physics import BatchedPhysics
from offline.segment import FutureContext


class ParallelRollout:
    """
    Handles H-step rollouts for candidate eval.

    Used during PGTO step to compare K candidates across R restarts.
    Uses deterministic (expected) physics for fair comparison.
    """

    def __init__(
        self, physics: BatchedPhysics, cmaes: CMAESModel, config: PGTOConfig
    ) -> None:
        self.physics = physics
        self.cmaes = cmaes
        self.config = config

    @torch.no_grad()
    def evaluate_candidates(
        self,
        history_states: torch.Tensor,  # [R*K, 20, 4]
        history_tokens: torch.Tensor,  # [R*K, 20]
        prev_lataccel: torch.Tensor,  # [R*K]
        prev_action: torch.Tensor,  # [R*K]
        cmaes_state: CMAESState,  # Batch size R*K
        future_context: FutureContext,  # {targets, roll, v_ego, a_ego} each [H]
        noise: torch.Tensor,  # [R*K, noise_window]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate R*K candidates over H-step horizon.

        Uses deterministic (expected) physics for fair comparison.

        Args:
            history_states: Current state history for all candidates
            history_tokens: Current token history for all candidates
            prev_lataccel: Current lataccel for all candidates
            cmaes_state: CMA-ES state for all candidates
            future_context: Future trajectory info
            noise: Noise to add for first noise_window steps

        Returns:
            costs: [R*K] total cost per candidate
            first_actions: [R*K] first action taken by each candidate
        """
        RK = history_states.shape[0]
        H = self.config.horizon

        # Clone for rollout
        states = history_states.clone()
        tokens = history_tokens.clone()
        cmaes_batch = cmaes_state.clone()
        prev_lat = prev_lataccel.clone()
        prev_action_h = prev_action.clone()

        total_tracking = torch.zeros(RK, device=self.config.device)
        total_jerk = torch.zeros(RK, device=self.config.device)
        total_action_smooth = torch.zeros(RK, device=self.config.device)  # Accumulate
        first_actions = None

        for h in range(H):
            target_h = future_context.targets[h]
            target_next = future_context.targets[h + 1]

            roll_h = future_context.roll[h]
            v_ego_h = future_context.v_ego[h]
            a_ego_h = future_context.a_ego[h]

            error_h = target_h - prev_lat

            # Update the error integral, skip first time step
            if h > 0:
                cmaes_batch.error_integral = torch.clamp(
                    cmaes_batch.error_integral + error_h, -5.0, 5.0
                )

            # Get CMA-ES action
            # Pass targets[h+1:] to match training code where 'future' starts at t+1
            features = self.cmaes.compute_features(
                target=target_h.expand(RK),
                current_lataccel=prev_lat,
                state=cmaes_batch,
                v_ego=v_ego_h.expand(RK),
                a_ego=a_ego_h.expand(RK),
                roll=roll_h.expand(RK),
                future_targets=future_context.targets[h + 1 :],
            )
            cmaes_actions = self.cmaes(features)

            # Add noise
            if h < self.config.noise_window:
                actions = cmaes_actions + noise[:, h]
            else:
                actions = cmaes_actions

            actions = actions.clamp(self.config.steer_min, self.config.steer_max)

            if first_actions is None:
                first_actions = actions.clone()

            total_action_smooth += (actions - prev_action_h) ** 2
            prev_action_h = actions

            # Update state history
            new_state = torch.stack(
                [
                    actions,
                    roll_h.expand(RK),
                    v_ego_h.expand(RK),
                    a_ego_h.expand(RK),
                ],
                dim=-1,
            )
            states = torch.cat([states[:, 1:, :], new_state.unsqueeze(1)], dim=1)

            # Deterministic physics step
            pred_lat = self.physics.expectation(states, tokens, prev_lat)

            # Costs (Compare pred at t+h+1 with target at t+h+1)
            tracking_error = pred_lat - target_next
            jerk = (pred_lat - prev_lat) / 0.1

            total_tracking += tracking_error**2
            total_jerk += jerk**2

            # Update CMA-ES state
            cmaes_batch.prev_error = error_h
            cmaes_batch.u_t2 = cmaes_batch.u_t1.clone()
            cmaes_batch.u_t1 = actions

            # Update tokens
            new_tokens = self.physics.tokenize(pred_lat)
            tokens = torch.cat([tokens[:, 1:], new_tokens.unsqueeze(1)], dim=1)

            prev_lat = pred_lat

        assert first_actions is not None

        # Total costs
        costs = (
            self.config.w_tracking * total_tracking
            + self.config.w_jerk * total_jerk
            + (10**self.config.w_action_smooth) * total_action_smooth
        )

        return costs, first_actions
