from dataclasses import dataclass


@dataclass
class PGTOConfig:
    """Configuration for PGTO optimization."""

    # Parallel restarts
    num_restarts: int = 3

    # Per-step optimization
    K: int = 2048  # Number of candidate trajectories per restart
    horizon: int = 6  # Lookahead horizon
    noise_window: int = 2  # Steps within horizon to inject noise
    noise_std: float = 0.15  # Noise standard deviation
    temperature: float = 0.5  # Temperature to combine candidate trajectories
    w_action_smooth = 500.0  # Action smoothing

    # Evaluation (match eval.py/tinyphysics.py)
    w_tracking: float = 5000.0
    w_jerk: float = 100.0
    context_length: int = 20
    control_start_idx: int = 100
    cost_steps: int = 400  # Only evaluate first 400 steps

    # Physics (match tinyphysics.py)
    physics_temperature: float = 0.8
    max_acc_delta: float = 0.5
    steer_min: float = -2.0
    steer_max: float = 2.0

    # Paths
    onnx_model_path: str = "models/tinyphysics.onnx"
    cmaes_params_path: str = "models/cmaes_params.npy"
    segments_dir: str = "data/"
    output_dir: str = "data/pgto/"

    # Device
    device: str = "cuda"


DEFAULT_CONFIG = PGTOConfig()
