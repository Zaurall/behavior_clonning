from dataclasses import dataclass


@dataclass
class PGTOConfig:
    """Configuration for PGTO optimization."""

    # Parallel restarts
    num_restarts: int = 10

    # Per-step optimization
    K: int = 128  # Number of candidate trajectories per iteration
    noise_window: int = 5  # Steps within horizon to inject noise
    n_iterations_max: int = 19
    elite_frac: float = 0.19
    horizon: int = 6
    noise_std_init: float = 0.19
    shift_threshold: float = 0.005
    w_action_smooth: float = 4.5  # Jerky action penalty
    w_variance: float = 1.0  # Linear multiplier to variance penalty (theoretically should just be 1 for risk-neutral)

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


@dataclass
class BCConfig:
    # Features
    context_length: int = 20  # Past history length
    future_length: int = 50  # Future context length
    control_start_idx: int = 100  # When control begins (from tinyphysics)

    # Model
    input_size: int = 247
    hidden_sizes: tuple[int, ...] = (1024, 1024, 512, 256, 128)

    # Training
    batch_size: int = 8192
    lr: float = 3e-4
    weight_decay: float = 1e-5
    epochs: int = 250

    # Evaluation
    eval_every_n_epochs: int = 50
    eval_num_segments: int = 500

    # Paths
    pgto_data_dir: str = "data/pgto/"
    segments_dir: str = "data/"
    output_dir: str = "models/"
    model_save_name: str = "bc_best.pt"

    # Device
    device: str = "cuda"
