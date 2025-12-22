from pathlib import Path

import optuna

from offline.config import PGTOConfig
from offline.pgto.optimizer import PGTOOptimizer
from offline.segment import load_segment


def objective(trial: optuna.Trial) -> float:
    """Test PGTO parameters on first file and return best cost."""

    # Sample hyperparameters
    params = {
        "horizon": trial.suggest_int("horizon", 6, 12),
        "noise_window": trial.suggest_int("noise_window", 1, 3),
        "noise_std": trial.suggest_float("noise_std", 0.01, 0.3, log=True),
        "temperature": trial.suggest_float("temperature", 0.001, 0.2, log=True),
        "w_action_smooth": trial.suggest_float("w_action_smooth", 4.0, 8.0),
    }

    print(f"Testing: {params}")

    try:
        # Create config with trial params
        config = PGTOConfig(
            num_restarts=3,
            K=2048,
            horizon=params["horizon"],
            noise_window=params["noise_window"],
            noise_std=params["noise_std"],
            temperature=params["temperature"],
            w_action_smooth=params["w_action_smooth"],
        )

        # Load segment and run optimizer
        segment = load_segment(Path("data/00000.csv"), config)
        optimizer = PGTOOptimizer(config)
        result = optimizer.optimize(segment, verbose=False)

        print(f"Best cost: {result.best_cost:.2f}")
        return result.best_cost

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
        study_name="pgto_hyperparameter_search_fixed",
        direction="minimize",
        load_if_exists=True,
    )

    study.optimize(objective)


if __name__ == "__main__":
    main()
