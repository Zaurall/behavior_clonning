import numpy as np
import torch

from controllers import BaseController


class Controller(BaseController):
    CONTEXT_LENGTH = 20
    FUTURE_LENGTH = 50
    INPUT_SIZE = 247
    CONTROL_START_IDX = 100

    def __init__(self, model_path: str = "models/bc_best.pt") -> None:
        self.model_path = model_path

        # Lazy load model on first update
        self.model = None
        self.device = None

        # Rolling buffers
        self.past_actions = np.zeros(self.CONTEXT_LENGTH, dtype=np.float32)
        self.past_lataccels = np.zeros(self.CONTEXT_LENGTH, dtype=np.float32)

        # Step tracking
        self.total_steps = 0  # Total calls to update()
        self.control_steps = 0  # Steps since control started
        self.segment_length = 400

    def _load_model(self) -> None:
        """Load model weights lazily."""
        from offline.bc.model import BCModel
        from offline.config import BCConfig

        self.device = torch.device("cpu")
        config = BCConfig()
        self.model = BCModel(config).to(self.device)
        self.model.load_state_dict(
            torch.load(self.model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

    @property
    def _in_control(self) -> bool:
        """Whether we're past warmup and in control."""
        # tinyphysics starts calling at step CONTEXT_LENGTH (20)
        # Control starts at CONTROL_START_IDX (100)
        # So after 80 calls, we're in control
        return self.total_steps >= (self.CONTROL_START_IDX - self.CONTEXT_LENGTH)

    def update(
        self,
        target_lataccel: float,
        current_lataccel: float,
        state,
        future_plan,
    ) -> float:
        if self.model is None:
            self._load_model()
            assert self.model is not None  # Typechecker

        # Build feature vector
        features = self._build_features(
            target_lataccel, current_lataccel, state, future_plan
        )

        # Forward pass
        with torch.no_grad():
            features_tensor = torch.tensor(
                features, dtype=torch.float32, device=self.device
            )
            action = self.model(features_tensor.unsqueeze(0)).item()

        # Update buffers and counters
        self._update_buffers(action, current_lataccel)
        self.total_steps += 1
        if self._in_control:
            self.control_steps += 1

        return action

    def _build_features(
        self,
        target_lataccel: float,
        current_lataccel: float,
        state,
        future_plan,
    ) -> np.ndarray:
        features = np.zeros(self.INPUT_SIZE, dtype=np.float32)

        # Current state [0:5]
        features[0] = target_lataccel
        features[1] = current_lataccel
        features[2] = state.roll_lataccel
        features[3] = state.v_ego
        features[4] = state.a_ego

        # Past actions [5:25] - zero out entries from before control
        past_actions = self.past_actions.copy()
        if self.control_steps < self.CONTEXT_LENGTH:
            num_to_zero = self.CONTEXT_LENGTH - self.control_steps
            past_actions[:num_to_zero] = 0.0
        features[5:25] = past_actions

        # Past lataccels [25:45]
        features[25:45] = self.past_lataccels

        # Future targets [45:95]
        features[45:95] = self._pad_future(future_plan.lataccel, target_lataccel)

        # Future roll [95:145]
        features[95:145] = self._pad_future(
            future_plan.roll_lataccel, state.roll_lataccel
        )

        # Future v_ego [145:195]
        features[145:195] = self._pad_future(future_plan.v_ego, state.v_ego)

        # Future a_ego [195:245]
        features[195:245] = self._pad_future(future_plan.a_ego, state.a_ego)

        # Timestep features [245:247]
        features[245] = (
            self.control_steps / self.segment_length
        )  # Segment progress (0 until control starts)
        features[246] = (
            min(self.control_steps, self.CONTEXT_LENGTH) / self.CONTEXT_LENGTH
        )  # Buffer validity

        return features

    def _pad_future(self, future_list: list, current_val: float) -> np.ndarray:
        arr = np.array(future_list, dtype=np.float32)
        if len(arr) >= self.FUTURE_LENGTH:
            return arr[: self.FUTURE_LENGTH]
        elif len(arr) > 0:
            pad_val = arr[-1]
            padding = np.full(self.FUTURE_LENGTH - len(arr), pad_val, dtype=np.float32)
            return np.concatenate([arr, padding])
        else:
            return np.full(self.FUTURE_LENGTH, current_val, dtype=np.float32)

    def _update_buffers(self, action: float, current_lataccel: float) -> None:
        self.past_actions = np.roll(self.past_actions, -1)
        self.past_actions[-1] = action

        self.past_lataccels = np.roll(self.past_lataccels, -1)
        self.past_lataccels[-1] = current_lataccel
