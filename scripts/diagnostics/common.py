"""Shared utilities for diagnostic experiments.

Provides device detection, tensor statistics, model loading, data preparation,
wandb helpers, and CLI argument parsing shared by pre/mid/post training scripts.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from deep_compressor.config import DeepCompressorConfig
from deep_compressor.data import NTPDataset, PaddingCollator, QADataset
from deep_compressor.model import DeepCompressor


# ─── device / tensor helpers (migrated from diagnostic.py) ───────────


def detect_device() -> torch.device:
    """Auto-detect best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def grad_norm(module: nn.Module) -> float:
    """Total L2 gradient norm across all parameters of *module*."""
    sq = 0.0
    for p in module.parameters():
        if p.grad is not None:
            sq += p.grad.data.norm(2).item() ** 2
    return math.sqrt(sq)


def stats_str(t: torch.Tensor) -> str:
    """One-line tensor statistics string."""
    with torch.no_grad():
        return (
            f"mean={t.mean().item():+.4f}  std={t.std().item():.4f}  "
            f"min={t.min().item():+.4f}  max={t.max().item():+.4f}"
        )


def to_device(batch: dict, device: torch.device) -> dict:
    """Move all tensor values in a batch dict to *device*."""
    return {
        k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()
    }


# ─── new statistical tools ───────────────────────────────────────────


def pairwise_cosine_similarity(x: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity matrix.

    Args:
        x: (N, D) tensor
    Returns:
        (N, N) cosine similarity matrix
    """
    x_norm = torch.nn.functional.normalize(x, dim=-1)
    return x_norm @ x_norm.T


def effective_rank(x: torch.Tensor) -> float:
    """Compute effective rank via SVD entropy method.

    The effective rank is exp(H) where H is the Shannon entropy of the
    normalized singular value distribution.

    Args:
        x: (N, D) tensor
    Returns:
        effective rank (float)
    """
    # SVD on the first sample if batched
    if x.dim() == 3:
        x = x[0]
    # SVD not supported on MPS — fall back to CPU
    s = torch.linalg.svdvals(x.float().cpu()).to(x.device)
    # Normalize to a probability distribution
    s = s / s.sum().clamp(min=1e-10)
    # Shannon entropy
    entropy = -(s * (s + 1e-10).log()).sum()
    return entropy.exp().item()


def gini_coefficient(values: torch.Tensor) -> float:
    """Compute Gini coefficient measuring concentration/inequality.

    Args:
        values: 1-D tensor of non-negative values
    Returns:
        Gini coefficient in [0, 1] (0 = perfect equality, 1 = maximum concentration)
    """
    values = values.float().flatten()
    if values.numel() == 0:
        return 0.0
    sorted_vals = torch.sort(values)[0]
    n = sorted_vals.numel()
    index = torch.arange(1, n + 1, device=values.device, dtype=torch.float)
    return ((2 * (index * sorted_vals).sum() / (n * sorted_vals.sum().clamp(min=1e-10)))
            - (n + 1) / n).item()


def attention_entropy(attn: torch.Tensor) -> torch.Tensor:
    """Compute entropy of attention distributions.

    Args:
        attn: (B, H, Q, S) attention probability tensor
    Returns:
        (B, H, Q) entropy per query position per head
    """
    eps = 1e-10
    return -(attn * (attn + eps).log()).sum(dim=-1)


# ─── model loading ───────────────────────────────────────────────────


def load_model(
    config: DeepCompressorConfig,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> DeepCompressor:
    """Build a DeepCompressor model, optionally loading checkpoint weights.

    Args:
        config: model configuration
        checkpoint_path: path to trainable_weights.pt (or directory containing it)
        device: target device (auto-detected if None)
    Returns:
        DeepCompressor model on *device*
    """
    if device is None:
        device = detect_device()

    model = DeepCompressor(config)

    if checkpoint_path is not None:
        # Accept either a .pt file or a directory containing trainable_weights.pt
        if os.path.isdir(checkpoint_path):
            checkpoint_path = os.path.join(checkpoint_path, "trainable_weights.pt")
        weights = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=False)
        print(f"  Loaded checkpoint from {checkpoint_path}")

    model.to(device)
    return model


# ─── data preparation ────────────────────────────────────────────────


def precompute_qwen_features(
    model: DeepCompressor,
    batch: dict,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pre-compute frozen Qwen document hidden states and mean-pooled features.

    Args:
        model: DeepCompressor on *device*
        batch: batch dict with doc_input_ids and doc_attention_mask on *device*
    Returns:
        (doc_hidden, doc_pooled) both detached tensors
    """
    with torch.no_grad():
        qwen_out = model.qwen(
            input_ids=batch["doc_input_ids"],
            attention_mask=batch["doc_attention_mask"],
            output_hidden_states=True,
            use_cache=False,
        )
        doc_hidden = qwen_out.hidden_states[-1].detach()
        mask_f = batch["doc_attention_mask"].unsqueeze(-1).float()
        doc_pooled = (
            (doc_hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        ).detach()
    del qwen_out
    return doc_hidden, doc_pooled


def prepare_ntp_batch(
    config: DeepCompressorConfig,
    data_path: str,
    batch_size: int,
    device: torch.device,
    min_doc_tokens: int = 0,
) -> Tuple[dict, AutoTokenizer]:
    """Load a single NTP batch and tokenizer for diagnostic use.

    Args:
        min_doc_tokens: Skip samples whose doc_input_ids is shorter than this.
            Useful for bottleneck experiments that need long documents.

    Returns:
        (batch_on_device, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(config.qwen.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = NTPDataset(
        data_path,
        tokenizer,
        max_doc_tokens=config.qwen.max_doc_tokens,
        segment_len=config.training.ntp_segment_len,
    )
    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)

    if min_doc_tokens > 0:
        # Select samples with long enough documents
        long_indices = []
        for i in range(len(ds)):
            sample = ds[i]
            doc_len = sample["doc_input_ids"].shape[0]
            if doc_len >= min_doc_tokens:
                long_indices.append(i)
            if len(long_indices) >= batch_size:
                break
        if not long_indices:
            print(f"  WARNING: no documents >= {min_doc_tokens} tokens, using first {batch_size}")
            long_indices = list(range(min(batch_size, len(ds))))
        else:
            print(f"  Found {len(long_indices)} docs >= {min_doc_tokens} tokens (indices: {long_indices})")
        n = len(long_indices)
        subset = Subset(ds, long_indices)
    else:
        n = min(batch_size, len(ds))
        subset = Subset(ds, list(range(n)))

    batch = next(
        iter(DataLoader(subset, batch_size=n, shuffle=False, collate_fn=collator))
    )
    batch = to_device(batch, device)
    return batch, tokenizer


def prepare_qa_loader(
    config: DeepCompressorConfig,
    data_path: str,
    batch_size: int,
    max_samples: int = 0,
) -> Tuple[DataLoader, AutoTokenizer]:
    """Build a QA DataLoader for evaluation diagnostics.

    Returns:
        (data_loader, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(config.qwen.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = QADataset(
        data_path,
        tokenizer,
        max_doc_tokens=config.qwen.max_doc_tokens,
        max_question_tokens=config.qwen.max_question_tokens,
        max_answer_tokens=config.qwen.max_answer_tokens,
    )
    if max_samples > 0 and max_samples < len(ds):
        ds = Subset(ds, list(range(max_samples)))

    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
    return loader, tokenizer


# ─── wandb helpers ────────────────────────────────────────────────────


def init_wandb(
    enabled: bool,
    project: str,
    run_name: str,
    config: Optional[dict] = None,
    entity: Optional[str] = None,
):
    """Initialize a wandb run if enabled.

    Returns:
        wandb run object or None
    """
    if not enabled:
        return None
    try:
        import wandb

        kwargs = dict(project=project, name=run_name)
        if config is not None:
            kwargs["config"] = config
        if entity is not None:
            kwargs["entity"] = entity
        return wandb.init(**kwargs)
    except ImportError:
        print("WARNING: wandb not installed, skipping wandb logging")
        return None


def log_wandb(run, metrics: dict, step: Optional[int] = None):
    """Log metrics to wandb if run is active."""
    if run is None:
        return
    if step is not None:
        run.log(metrics, step=step)
    else:
        run.log(metrics)


def finish_wandb(run):
    """Finish wandb run if active."""
    if run is not None:
        run.finish()


# ─── CLI factory ──────────────────────────────────────────────────────


def base_parser(description: str) -> argparse.ArgumentParser:
    """Create a base argument parser with common diagnostic arguments."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/macbook_debug.yaml",
                        help="Path to YAML config file")
    parser.add_argument("--data_path", default="data/ntp_tiny.jsonl",
                        help="Path to NTP data file")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="dc-diagnostic",
                        help="wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="wandb entity (team/user)")
    return parser
