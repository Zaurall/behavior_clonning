import onnx2torch
import torch
import torch.nn as nn
import torch.nn.functional as F

from offline.config import PGTOConfig


class BatchedPhysics(nn.Module):
    """
    Batched physics model (to internally simulate tinyphysics.py).
    Has both stochastic (same as tinyphysics) and differentiable expectation (for gradient dissent).
    """

    bins: torch.Tensor

    def __init__(self, config: PGTOConfig) -> None:
        super().__init__()

        self.config = config

        self.model = (
            onnx2torch.convert(self.config.onnx_model_path)
            .to(self.config.device)
            .eval()
        )

        # Bins for tokenization
        self.register_buffer(
            "bins", torch.linspace(-5, 5, 1024, device=self.config.device)
        )

    @torch.no_grad()
    def forward(
        self,
        states: torch.Tensor,
        tokens: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass through the physics model.

        Args:
            states: [B, 20, 4] - history of [action, roll, v_ego, a_ego]
            tokens: [B, 20] - tokenized lataccel history

        Returns:
            logits: [B, 1024] - logits over lataccel bins
        """
        return self.model(states, tokens)

    @torch.no_grad()
    def sample(
        self,
        states: torch.Tensor,
        tokens: torch.Tensor,
        prev_lataccel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample lataccel from the physics model (stochastic, matches tinyphysics).

        Args:
            states: [B, 20, 4]
            tokens: [B, 20]
            prev_lataccel: [B] - used for clamping

        Returns:
            lataccel: [B] - sampled and clamped lataccel
        """
        logits = self.forward(states, tokens)
        probs = F.softmax(logits / self.config.temperature, dim=-1)

        # Sample from distribution
        indices = torch.multinomial(probs, num_samples=1).squeeze(-1)
        lataccel = self.bins[indices]

        # Clamp
        lataccel = torch.clamp(
            lataccel,
            prev_lataccel - self.config.max_acc_delta,
            prev_lataccel + self.config.max_acc_delta,
        )

        return lataccel

    @torch.no_grad()
    def expectation(
        self,
        states: torch.Tensor,
        tokens: torch.Tensor,
        prev_lataccel: torch.Tensor,
    ) -> torch.Tensor:
        """
        Deterministic expectation (for efficient candidate comparison).


        Args:
            states: [B, 20, 4]
            tokens: [B, 20]
            prev_lataccel: [B] - used for clamping

        Returns:
            lataccel: [B] - expected value, clamped
        """
        logits = self.forward(states, tokens)
        probs = F.softmax(logits / self.config.temperature, dim=-1)
        lataccel = (probs * self.bins).sum(dim=-1)

        # Clamp
        lataccel = torch.clamp(
            lataccel,
            prev_lataccel - self.config.max_acc_delta,
            prev_lataccel + self.config.max_acc_delta,
        )

        return lataccel
