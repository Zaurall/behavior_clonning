import torch
import torch.nn as nn
import numpy as np
from offline.config import PGTOConfig, BCConfig
from offline.bc.model import BCModel
from offline.cmaes import CMAESState
from offline.segment import FutureContext

class BCPrior(nn.Module):
    def __init__(self, config: PGTOConfig):
        super().__init__()
        self.config = config
        self.device = config.device
        
        # Load BC Model
        bc_config = BCConfig() # default config for model architecture, assumes it matches saved model
        self.model = BCModel(bc_config)
        
        # Load weights
        print(f"Loading BC Prior from {config.bc_model_path}")
        state_dict = torch.load(config.bc_model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # For decoding tokens if needed (but we receive values usually)
        # We might need physics instance to decode tokens if passed
        self.physics = None 

    def set_physics(self, physics):
        self.physics = physics

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)
        
    def compute_features(
        self,
        target: torch.Tensor,
        current_lataccel: torch.Tensor,
        state: CMAESState,
        v_ego: torch.Tensor,
        a_ego: torch.Tensor,
        roll: torch.Tensor,
        future_targets: torch.Tensor, # legacy arg from cmaes
        # New args
        history_states: torch.Tensor, # [B, 20, 4]
        history_lataccel: torch.Tensor, # [B, 20]
        future_context: FutureContext,
        h: int, # relative step in rollout
        t: int, # absolute step in segment
        **kwargs
    ) -> torch.Tensor:
        """
        Construct features for BC Model.
        """
        B = target.shape[0]
        
        # 1. Current State [0:5]
        # target, current_lat, roll, v, a
        current_features = torch.stack([
            target,
            current_lataccel,
            roll,
            v_ego,
            a_ego
        ], dim=-1) # [B, 5]
        
        # 2. Past Actions [5:25]
        # history_states: [B, 20, 4] -> index 0 is action
        # The BC model in training zeroed out actions before control start.
        # Here we rely on the caller passing correct history.
        # history_states[:, :, 0] is actions.
        # history_states is oldest -> newest.
        past_actions = history_states[:, :, 0] # [B, 20]
        
        # In eval/training, we might need to handle the "zeroing out" before control starts.
        # If t < control_start, we might have partial history.
        # But in PGTO, we assume we are optimizing FORWARD.
        # If t < 100, we should preserve zeros if they were zero.
        # The history passed in from PGTO comes from `segment.initial_history_states` or rollout updates.
        # We'll assume history_states is correct.
        
        # 3. Past Lataccels [25:45]
        # passed as history_lataccel [B, 20]
        past_lataccels = history_lataccel
        
        # 4. Future Plan [45:245]
        # BC expects 50 steps of future.
        # future_plan consists of: lataccel (target), roll, v_ego, a_ego
        # future_context provided has tensors starting from t (start of rollout)
        # We need future from t+h.
        
        future_idx_start = h
        future_len_req = 50
        
        # Helper to extract and pad
        def extract_future(source_tensor: torch.Tensor, current_val: torch.Tensor) -> torch.Tensor:
            # source_tensor is [H_total] (shared) or [H_total, B?] No, FutureContext has [H_total] usually?
            # BUT in ParallelRollout, we might need to expand them if they are not expanded?
            # Wait, FutureContext in ParallelRollout has tensors of shape [H]. 
            # Optimizer gets future_context from Segment which returns shape [horizon + lookahead].
            # So source_tensor is [horizon + lookahead].
            
            # We want source_tensor[future_idx_start : future_idx_start + 50]
            # Since source_tensor is 1D (shared across batch), we slice it, then repeat for batch.
            
            avail_len = source_tensor.shape[0]
            start = future_idx_start
            end = min(start + future_len_req, avail_len)
            
            # Slice
            slice_data = source_tensor[start:end] # [L]
            
            # Pad if needed
            L = slice_data.shape[0]
            if L < future_len_req:
                pad_len = future_len_req - L
                if L > 0:
                    val = slice_data[-1]
                else:
                    # Fallback if empty future? Should not happen if lookahead is handled
                    # But if it is empty, use current_val. 
                    # Warning: current_val is [B], but slice_data is scalar (1D). 
                    # We should probably expand slice_data first?
                    # Actually, if we pad with scalar, that's fine.
                    val = current_val[0] if isinstance(current_val, torch.Tensor) and current_val.ndim>0 else 0.0 # Hacky
                    
                padding = val.expand(pad_len)
                slice_data = torch.cat([slice_data, padding])
                
            # Expand to batch
            return slice_data.unsqueeze(0).expand(B, -1) # [B, 50]

        # Extract futures
        # Note: future_context.targets is NOT expanded in rollout (it's shared).
        # But target, roll, etc passed as args ARE expanded.
        # We should use future_context directly.
        
        f_targets = extract_future(future_context.targets, target)
        f_roll = extract_future(future_context.roll, roll)
        f_v = extract_future(future_context.v_ego, v_ego)
        f_a = extract_future(future_context.a_ego, a_ego)
        
        future_features = torch.cat([f_targets, f_roll, f_v, f_a], dim=1) # [B, 200]
        
        # 5. Timestep Features [245:247]
        # progress = control_steps / segment_length (400)
        # valid = min(control_steps, 20) / 20
        
        control_steps = (t + h) - self.config.control_start_idx
        control_steps = max(0, control_steps) # Clamp at 0
        
        progress = control_steps / 400.0
        validity = min(control_steps, 20.0) / 20.0
        
        time_features = torch.stack([
            torch.full((B,), progress, device=self.device),
            torch.full((B,), validity, device=self.device)
        ], dim=-1)
        
        # Combine all
        features = torch.cat([
            current_features,
            past_actions,
            past_lataccels,
            future_features,
            time_features
        ], dim=1)
        
        return features
