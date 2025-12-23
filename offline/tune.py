from pathlib import Path

import numpy as np
import optuna

from offline.config import PGTOConfig
from offline.pgto.optimizer import PGTOOptimizer
from offline.segment import load_segment


def objective(trial: optuna.Trial) -> float:
    """Test PGTO parameters on representative files and return mean cost."""
    SEGMENTS = [
        "00432",
        "03061",
        "03935",
        "02472",
        "04699",
        "02896",
        "04185",
        "02591",
        "00692",
        "00219",
    ]

    params = {
        "horizon": trial.suggest_int("horizon", 5, 16),
        "noise_window": trial.suggest_int("noise_window", 1, 2),
        "noise_std": trial.suggest_float("noise_std", 0.01, 0.8, log=True),
        "w_action_smooth": trial.suggest_float("w_action_smooth", 4.0, 9.0),
    }
    print(f"Testing: {params}")

    try:
        config = PGTOConfig(
            num_restarts=3,
            K=2048,
            horizon=params["horizon"],
            noise_window=params["noise_window"],
            noise_std=params["noise_std"],
            w_action_smooth=params["w_action_smooth"],
        )
        optimizer = PGTOOptimizer(config)

        total = 0.0
        for i, seg_id in enumerate(SEGMENTS):
            segment = load_segment(Path(f"data/{seg_id}.csv"), config)
            result = optimizer.optimize(segment, verbose=False)
            total += float(np.mean([r.cost for r in result.restarts]))

            # Report running average
            avg_so_far = total / (i + 1)
            trial.report(avg_so_far, step=i)
            print(f"  Segment {seg_id}: avg so far = {avg_so_far:.2f}")

            if trial.should_prune():
                raise optuna.TrialPruned()

        return total / len(SEGMENTS)

    except optuna.TrialPruned:
        raise
    except Exception as e:
        print(f"Trial failed: {e}")
        raise optuna.TrialPruned()


def main():
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=50,
        multivariate=True,
        group=True,
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=25,
        n_warmup_steps=1,
        interval_steps=1,
    )

    study = optuna.create_study(
        sampler=sampler,
        pruner=pruner,
        storage="sqlite:///optuna_pgto_study.db",
        study_name="pgto_hyperparameter_search_representative",
        direction="minimize",
        load_if_exists=True,
    )

    study.optimize(objective)


if __name__ == "__main__":
    main()
