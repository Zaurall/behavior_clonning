from dataclasses import dataclass


@dataclass
class PGTOConfig:
    """Configuration for PGTO optimization."""

    # Parallel restarts
    num_restarts: int = 3

    # Per-step optimization
    K: int = 1024  # Number of candidate trajectories per restart
    horizon: int = 10  # Lookahead horizon
    noise_window: int = 2  # Steps within horizon to inject noise
    noise_std: float = 0.1  # Noise standard deviation
    temperature: float = 2.0  # Temperature to combine candidate trajectories

    # Internal evaluation
    eval_mc_samples: int = 20  # MC samples (per restart) for final evaluation

    # External evaluation (match eval.py/tinyphysics.py)
    w_tracking: float = 100.0
    w_jerk: float = 100.0
    context_length: int = 20
    control_start_idx: int = 100

    # Physics (match tinyphysics.py)
    onnx_model_path: str = "models/tinyphysics.onnx"
    physics_temperature: float = 0.8
    max_acc_delta: float = 0.5
    steer_min: float = -2.0
    steer_max: float = 2.0

    # Device
    device: str = "mps"


DEFAULT_CONFIG = PGTOConfig()
