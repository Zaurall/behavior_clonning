import time
from pathlib import Path

import numpy as np
import optuna

from offline.config import PGTOConfig
from offline.pgto.optimizer import PGTOOptimizer
from offline.segment import load_segment


def objective(trial: optuna.Trial) -> tuple[float, float]:
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
        "K": trial.suggest_categorical("K", [128, 256, 512, 768, 1024]),
        "n_iterations_max": trial.suggest_int("n_iterations_max", 2, 5),
        "elite_frac": trial.suggest_float("elite_frac", 0.02, 0.2),
        "horizon_init": trial.suggest_int("horizon_init", 4, 10),
        "horizon_scale": trial.suggest_float("horizon_scale", 1.0, 1.3),
        "noise_window": trial.suggest_int("noise_window", 1, 4),
        "noise_std_init": trial.suggest_float("noise_std_init", 0.02, 0.20, log=True),
        "w_action_smooth": trial.suggest_float("w_action_smooth", 3.5, 6.0),
        "shift_threshold": trial.suggest_float(
            "shift_threshold", 0.001, 0.05, log=True
        ),
    }
    print(f"Testing: {params}")

    config = PGTOConfig(
        num_restarts=3,
        K=params["K"],
        n_iterations_max=params["n_iterations_max"],
        elite_frac=params["elite_frac"],
        horizon_init=params["horizon_init"],
        horizon_scale=params["horizon_scale"],
        noise_window=params["noise_window"],
        noise_std_init=params["noise_std_init"],
        w_action_smooth=params["w_action_smooth"],
        shift_threshold=params["shift_threshold"],
    )
    optimizer = PGTOOptimizer(config)

    total_cost = 0.0
    total_time = 0.0

    for seg_id in SEGMENTS:
        segment = load_segment(Path(f"data/{seg_id}.csv"), config)

        start = time.time()
        result = optimizer.optimize(segment, verbose=False)
        elapsed = time.time() - start

        total_cost += float(np.mean([r.cost for r in result.restarts]))
        total_time += elapsed

    avg_cost = total_cost / len(SEGMENTS)
    avg_time = total_time / len(SEGMENTS)

    return avg_cost, avg_time


def main():
    sampler = optuna.samplers.TPESampler(
        n_startup_trials=25,
        multivariate=True,
        constant_liar=True,
    )

    study = optuna.create_study(
        sampler=sampler,
        storage="sqlite:///optuna_pgto_study.db",
        study_name="pgto_hyperparameter_search_multi_2",
        directions=["minimize", "minimize"],
        load_if_exists=True,
    )

    study.optimize(objective)


if __name__ == "__main__":
    main()
