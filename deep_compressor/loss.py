"""Distillation and total loss computation for Deep Compressor."""

from typing import Dict, List, Optional

import torch
import torch.nn.functional as F


class DistillationLoss:
    """Computes KL divergence and hidden-state MSE losses for teacher distillation."""

    def __init__(self, temperature: float = 2.0, hidden_distill_layers: Optional[List[int]] = None,
                 hidden_distill_ramp_steps: int = 2000):
        self.temperature = temperature
        self.hidden_distill_layers = hidden_distill_layers or [7, 14, 21, 27]
        self.hidden_distill_ramp_steps = hidden_distill_ramp_steps

    def compute_kl_loss(
        self, student_logits: torch.Tensor, teacher_logits: torch.Tensor,
        answer_mask: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence on answer token positions with temperature scaling.

        Args:
            student_logits: (batch, seq_len, vocab_size)
            teacher_logits: (batch, seq_len, vocab_size)
            answer_mask: (batch, seq_len) — True for answer token positions
        Returns:
            scalar KL loss
        """
        T = self.temperature
        student_log_probs = F.log_softmax(student_logits / T, dim=-1)
        teacher_probs = F.softmax(teacher_logits / T, dim=-1)

        # KL(teacher || student) = sum(teacher * (log(teacher) - log(student)))
        kl = F.kl_div(student_log_probs, teacher_probs, reduction="none").sum(dim=-1)  # (B, S)
        kl = kl * answer_mask
        # Scale by T^2 as per distillation convention
        num_tokens = answer_mask.sum().clamp(min=1)
        return (kl.sum() / num_tokens) * (T * T)

    def compute_hidden_mse_loss(
        self, student_hidden: List[torch.Tensor], teacher_hidden: List[torch.Tensor],
        shared_mask: torch.Tensor, global_step: int,
    ) -> torch.Tensor:
        """MSE loss between student and teacher hidden states at selected layers.

        Only computed at shared question+answer token positions. Weight ramps linearly.

        Args:
            student_hidden: list of (batch, student_seq_len, hidden_dim) per layer
            teacher_hidden: list of (batch, teacher_seq_len, hidden_dim) per layer
            shared_mask: (batch, shared_len) — mask for valid shared positions
            global_step: current training step for ramp computation
        Returns:
            scalar MSE loss
        """
        ramp = min(1.0, global_step / max(1, self.hidden_distill_ramp_steps))

        total = torch.tensor(0.0, device=student_hidden[0].device)
        count = 0
        for layer_idx in self.hidden_distill_layers:
            s_h = student_hidden[layer_idx]  # (B, shared_len, D)
            t_h = teacher_hidden[layer_idx]  # (B, shared_len, D)
            # MSE per position
            mse = (s_h - t_h).pow(2).mean(dim=-1)  # (B, shared_len)
            mse = mse * shared_mask
            num_tokens = shared_mask.sum().clamp(min=1)
            total = total + mse.sum() / num_tokens
            count += 1

        if count == 0:
            return total
        return (total / count) * ramp


def compute_total_loss(
    qa_ce_loss: torch.Tensor,
    kl_loss: Optional[torch.Tensor] = None,
    hidden_mse_loss: Optional[torch.Tensor] = None,
    qa_ce_weight: float = 1.0,
    kl_weight: float = 1.0,
    hidden_mse_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Combine all loss components with weights.

    Returns dict with 'total' and individual components.
    """
    total = qa_ce_weight * qa_ce_loss
    components = {"qa_ce": qa_ce_loss}

    if kl_loss is not None:
        total = total + kl_weight * kl_loss
        components["kl"] = kl_loss
    if hidden_mse_loss is not None:
        total = total + hidden_mse_weight * hidden_mse_loss
        components["hidden_mse"] = hidden_mse_loss

    components["total"] = total
    return components
