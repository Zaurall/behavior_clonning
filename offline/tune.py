from pathlib import Path

import numpy as np
import optuna

from offline.config import PGTOConfig
from offline.pgto.optimizer import PGTOOptimizer
from offline.segment import load_segment


def objective(trial: optuna.Trial) -> float:
    """Test PGTO parameters on first file and return best cost."""

    # Sample hyperparameters
    params = {
        "horizon": trial.suggest_int("horizon", 6, 12),
        "noise_window": trial.suggest_int("noise_window", 1, 4),
        "noise_std": trial.suggest_float("noise_std", 0.01, 0.3, log=True),
        "w_action_smooth": trial.suggest_float("w_action_smooth", 4.0, 8.0),
    }

    print(f"Testing: {params}")

    try:
        # Create config with trial params
        config = PGTOConfig(
            num_restarts=5,
            K=2048,
            horizon=params["horizon"],
            noise_window=params["noise_window"],
            noise_std=params["noise_std"],
            w_action_smooth=params["w_action_smooth"],
        )

        optimizer = PGTOOptimizer(config)

        total = 0.0
        for i in range(5):  # First 5 segments
            segment = load_segment(Path(f"data/{i:05d}.csv"), config)
            result = optimizer.optimize(segment, verbose=False)
            total += float(np.mean([r.cost for r in result.restarts]))

        return total / 5

    except Exception as e:
        print(f"Trial failed: {e}")
        raise optuna.TrialPruned()


def main():
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=30,
        multivariate=True,
        group=True,
    )

    study = optuna.create_study(
        sampler=sampler,
        storage="sqlite:///optuna_pgto_study.db",
        study_name="pgto_hyperparameter_search_5",
        direction="minimize",
        load_if_exists=True,
    )

    study.optimize(objective)


if __name__ == "__main__":
    main()
