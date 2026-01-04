import os

os.environ["OMP_NUM_THREADS"] = "1"  # Prevent deadlock on SLURM cluster

from functools import partial
from pathlib import Path

import numpy as np
from tqdm.contrib.concurrent import process_map

from offline.config import BCConfig


def _run_segment(seg_path, model_path):
    from controllers.bc import Controller
    from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

    physics = TinyPhysicsModel("models/tinyphysics.onnx", debug=False)
    controller = Controller(model_path=model_path)
    sim = TinyPhysicsSimulator(
        physics, str(seg_path), controller=controller, debug=False
    )
    return sim.rollout()["total_cost"]


def evaluate_online(model_path: Path, num_segments: int, config: BCConfig) -> float:
    segment_paths = sorted(Path(config.segments_dir).glob("*.csv"))[:num_segments]

    run_fn = partial(_run_segment, model_path=str(model_path))
    costs = process_map(
        run_fn, segment_paths, max_workers=30, chunksize=10, desc="Eval"
    )

    return float(np.mean(costs))
