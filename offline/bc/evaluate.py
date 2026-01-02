from pathlib import Path

import numpy as np
from tqdm import tqdm

from offline.config import BCConfig


def evaluate_online(
    model_path: Path,
    num_segments: int,
    config: BCConfig,
) -> float:
    """
    Evaluate BC model on multiple segments using tinyphysics.

    Runs sequentially to avoid multiprocessing issues with model loading.
    """
    # Import here to avoid circular imports
    from controllers.bc import Controller
    from tinyphysics import TinyPhysicsModel, TinyPhysicsSimulator

    segments_dir = Path(config.segments_dir)
    segment_paths = sorted(segments_dir.glob("*.csv"))[:num_segments]

    physics_model = TinyPhysicsModel("models/tinyphysics.onnx", debug=False)
    costs = []

    for seg_path in tqdm(segment_paths, desc="Evaluating", leave=False):
        controller = Controller(model_path=str(model_path))
        sim = TinyPhysicsSimulator(
            physics_model, str(seg_path), controller=controller, debug=False
        )
        result = sim.rollout()
        costs.append(result["total_cost"])

    return float(np.mean(costs))
