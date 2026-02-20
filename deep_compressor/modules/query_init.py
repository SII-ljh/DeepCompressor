"""QueryInit: learnable base queries with question-conditioned additive bias."""

import torch
import torch.nn as nn


class QueryInit(nn.Module):
    def __init__(self, num_queries: int, perceiver_dim: int, qwen_dim: int):
        super().__init__()
        self.base_queries = nn.Parameter(torch.randn(num_queries, perceiver_dim) * 0.02)
        self.question_proj = nn.Linear(qwen_dim, perceiver_dim)

    def forward(self, question_pooled: torch.Tensor) -> torch.Tensor:
        """
        Args:
            question_pooled: (batch, qwen_dim) — mean-pooled question hidden state
        Returns:
            (batch, num_queries, perceiver_dim) — initial query vectors
        """
        bias = self.question_proj(question_pooled).unsqueeze(1)  # (batch, 1, perceiver_dim)
        queries = self.base_queries.unsqueeze(0) + bias  # (batch, num_queries, perceiver_dim)
        return queries
