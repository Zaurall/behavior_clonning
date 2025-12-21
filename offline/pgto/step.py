import torch
import torch.nn.functional as F

from offline.cmaes import CMAESModel, CMAESState
from offline.config import PGTOConfig
from offline.pgto.rollout import ParallelRollout
from offline.physics import BatchedPhysics
from offline.segment import FutureContext


class PGTOStep:
    """
    Single timestamp PGTO optimization with multiple (parallel) restarts.

    For R restarts, sample K candidates each, R*K total evals,
    returns 1 best action per restart (R actions).
    """

    def __init__(
        self,
        physics: BatchedPhysics,
        cmaes: CMAESModel,
        config: PGTOConfig,
    ) -> None:
        self.physics = physics
        self.cmaes = cmaes
        self.config = config

        self.rollout = ParallelRollout(physics=physics, cmaes=cmaes, config=config)

    @torch.no_grad()
    def optimize(
        self,
        history_states: torch.Tensor,  # [R, 20, 4]
        history_tokens: torch.Tensor,  # [R, 20]
        prev_lataccel: torch.Tensor,  # [R]
        cmaes_state: CMAESState,  # Batch size R
        future_context: FutureContext,  # {targets, roll, v_ego, a_ego} each [H]
    ) -> torch.Tensor:
        """
        Find optimal action for each of R restarts.

        Args:
            history_states: State history for each restart
            history_tokens: Token history for each restart
            prev_lataccel: Current lataccel for each restart
            cmaes_state: CMA-ES state for each restart
            future_context: Future trajectory context (shared across restarts)

        Returns:
            best_actions: [R] optimal action for each restart

        """
        R, K = self.config.num_restarts, self.config.K
        RK = R * K

        # Expand states from R to R*K (each restart has K candidates)
        states_expanded = history_states.repeat_interleave(K, dim=0)  # [R*K, 20, 4]
        tokens_expanded = history_tokens.repeat_interleave(K, dim=0)  # [R*K, 20]
        prev_lat_expanded = prev_lataccel.repeat_interleave(K)  # [R*K]
        cmaes_expanded = cmaes_state.expand(RK)

        # Antithetic noise: generate half, mirror within each restart
        half_noise = (
            torch.randn(R, K // 2, self.config.noise_window, device=self.config.device)
            * self.config.noise_std
        )
        noise = torch.cat([half_noise, -half_noise], dim=1)  # [R, K, noise_window]
        noise = noise.view(RK, self.config.noise_window)  # [R*K, noise_window]

        # Evaluate all candidates
        costs, first_actions = self.rollout.evaluate_candidates(
            history_states=states_expanded,
            history_tokens=tokens_expanded,
            prev_lataccel=prev_lat_expanded,
            cmaes_state=cmaes_expanded,
            future_context=future_context,
            noise=noise,
        )

        # Reshape to [R, K]
        costs = costs.view(R, K)
        first_actions = first_actions.view(R, K)

        # Importance weight based on temperature
        weights = F.softmax(-costs / self.config.temperature, dim=1)  # [R, K]

        best_actions = (weights * first_actions).sum(dim=1)  # [R]

        return best_actions
