import torch

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
        prev_action: torch.Tensor,  # [R]
        cmaes_state: CMAESState,  # Batch size R
        future_context: FutureContext,  # {targets, roll, v_ego, a_ego} each [H]
    ) -> tuple[torch.Tensor, int]:
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
        R = self.config.num_restarts
        K = self.config.K

        assert K % 2 == 0, "K must be even for antithetic sampling"
        half_K = K // 2

        RK = R * K

        best_actions = None
        best_costs = torch.full((R,), float("inf"), device=self.config.device)

        mean = torch.zeros(R, self.config.noise_window, device=self.config.device)
        std = torch.full(
            (R, self.config.noise_window),
            self.config.noise_std_init,
            device=self.config.device,
        )

        prev_mean = mean.clone()

        iteration = None  # for type checker
        for iteration in range(self.config.n_iterations_max):
            horizon = int(
                self.config.horizon_init * (self.config.horizon_scale**iteration)
            )

            # Expand states from R to R*K (each restart has K candidates)
            states_expanded = history_states.repeat_interleave(K, dim=0)  # [R*K, 20, 4]
            tokens_expanded = history_tokens.repeat_interleave(K, dim=0)  # [R*K, 20]
            prev_lat_expanded = prev_lataccel.repeat_interleave(K)  # [R*K]
            prev_action_expanded = prev_action.repeat_interleave(K)  # [R*K]
            cmaes_expanded = cmaes_state.expand(RK)

            # Antithetic noise: generate half, mirror within each restart
            half_noise = torch.randn(
                R, half_K, self.config.noise_window, device=self.config.device
            ) * std.unsqueeze(1)
            noise = torch.cat([half_noise, -half_noise], dim=1)  # [R, K, noise_window]

            # Add mean offset to get candidates
            candidates = mean.unsqueeze(1) + noise  # [R, K, noise_window]

            noise_flat = candidates.view(RK, self.config.noise_window)

            # Evaluate all candidates
            costs, first_actions = self.rollout.evaluate_candidates(
                history_states=states_expanded,
                history_tokens=tokens_expanded,
                prev_lataccel=prev_lat_expanded,
                prev_action=prev_action_expanded,
                cmaes_state=cmaes_expanded,
                future_context=future_context,
                noise=noise_flat,
                horizon=horizon,
            )

            # Reshape to [R, K]
            costs = costs.view(R, K)
            first_actions = first_actions.view(R, K)
            candidates = candidates.view(R, K, self.config.noise_window)

            # Track best ever across all iterations
            iter_best_costs, iter_best_idx = costs.min(dim=1)
            iter_best_actions = first_actions.gather(
                1, iter_best_idx.unsqueeze(1)
            ).squeeze(1)

            improved = iter_best_costs < best_costs
            best_costs = torch.where(improved, iter_best_costs, best_costs)
            if best_actions is None:
                best_actions = iter_best_actions
            else:
                best_actions = torch.where(improved, iter_best_actions, best_actions)

            # Select elites
            n_elite = max(1, int(K * self.config.elite_frac))
            elite_idx = costs.argsort(dim=1)[:, :n_elite]  # [R, n_elite]

            elite_candidates = candidates.gather(
                1, elite_idx.unsqueeze(-1).expand(-1, -1, self.config.noise_window)
            )

            prev_mean = mean.clone()
            mean = elite_candidates.mean(dim=1)  # [R, noise_window]
            std = elite_candidates.std(dim=1).clamp(min=0.01)  # [R, noise_window]

            # Early termination
            if iteration > 0:
                shift = (mean - prev_mean).norm(dim=1).max()
                if shift < self.config.shift_threshold:
                    break

        assert best_actions is not None
        assert iteration is not None

        return best_actions, iteration + 1
