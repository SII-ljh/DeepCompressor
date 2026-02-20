"""NERHead: produces per-token entity probability scores from FinBERT hidden states."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NERHead(nn.Module):
    def __init__(self, finbert_dim: int, num_labels: int = 15):
        """
        Args:
            finbert_dim: FinBERT hidden size
            num_labels: BIO tags count (7 entity types * 2 + O = 15)
        """
        super().__init__()
        self.classifier = nn.Linear(finbert_dim, num_labels)
        # Label 0 is the "O" (outside) tag
        self.o_label_idx = 0

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: (batch, seq_len, finbert_dim)
        Returns:
            entity_scores: (batch, seq_len) — probability of being an entity ∈ [0, 1]
        """
        logits = self.classifier(hidden_states)  # (B, S, num_labels)
        probs = F.softmax(logits, dim=-1)
        # Entity score = 1 - P(O)
        entity_scores = 1.0 - probs[:, :, self.o_label_idx]
        return entity_scores
