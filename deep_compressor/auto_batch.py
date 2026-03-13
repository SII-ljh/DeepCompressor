"""Auto batch size detection and gradient accumulation computation."""

from __future__ import annotations

import gc
import logging
import math
import warnings
from typing import Callable

import torch

logger = logging.getLogger(__name__)


def find_max_batch_size(
    model: torch.nn.Module,
    build_batch_fn: Callable[[int], dict],
    forward_fn: Callable[[torch.nn.Module, dict], torch.Tensor],
    device: torch.device,
    max_batch: int = 64,
    min_batch: int = 1,
    mixed_precision: str = "no",
) -> int:
    """Find maximum batch size via descending power-of-2 probe.

    On MPS/CPU, returns a conservative default (4) with a warning.

    Args:
        model: Model to probe (should already be on *device*).
        build_batch_fn: ``fn(batch_size) -> dict`` of tensors on *device*.
        forward_fn: ``fn(model, batch) -> loss``.  Should run forward only
            (backward is handled here).
        device: Device to probe on.
        max_batch: Upper bound for batch size (default 64).
        min_batch: Lower bound (default 1).
        mixed_precision: ``"no"``, ``"fp16"``, or ``"bf16"`` — mirrors
            training autocast settings.

    Returns:
        Maximum power-of-2 batch size that fits in GPU memory.
    """
    if device.type not in ("cuda",):
        warnings.warn(
            f"Auto batch size detection not supported on {device.type}. "
            f"Returning conservative default of 4. "
            f"Set batch_size explicitly for best performance.",
            stacklevel=2,
        )
        return 4

    # Highest power of 2 <= max_batch
    batch_size = 1
    while batch_size * 2 <= max_batch:
        batch_size *= 2

    amp_dtype = None
    if mixed_precision == "fp16":
        amp_dtype = torch.float16
    elif mixed_precision == "bf16":
        amp_dtype = torch.bfloat16

    best = min_batch

    while batch_size >= min_batch:
        try:
            _cleanup_gpu()
            batch = build_batch_fn(batch_size)

            if amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    loss = forward_fn(model, batch)
            else:
                loss = forward_fn(model, batch)

            loss.backward()

            # Success — this batch size fits
            best = batch_size
            logger.info(f"Auto batch size: {batch_size} fits in memory")

            model.zero_grad(set_to_none=True)
            del loss, batch
            _cleanup_gpu()
            break

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() and isinstance(e, RuntimeError):
                raise  # Re-raise non-OOM RuntimeErrors

            logger.info(f"Auto batch size: {batch_size} OOM, trying smaller")
            model.zero_grad(set_to_none=True)
            _cleanup_gpu()
            batch_size //= 2

    if best < min_batch:
        best = min_batch

    logger.info(f"Auto batch size detection result: {best}")
    return best


def _cleanup_gpu():
    """Release GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def compute_gradient_accumulation(
    per_gpu_batch: int,
    num_gpus: int,
    target: int = 256,
) -> int:
    """Compute gradient accumulation steps to reach *target* effective batch.

    ``effective = per_gpu_batch * num_gpus * grad_accum``

    Returns at least 1.
    """
    per_step = per_gpu_batch * num_gpus
    return max(1, math.ceil(target / per_step))


def build_sample_batch_fn_qa(config, vocab_size: int, device: torch.device):
    """Return a closure that builds synthetic QA batches for memory probing.

    Uses worst-case (max) sequence lengths from *config* for conservative
    estimation.
    """
    max_doc = config.qwen.max_doc_tokens
    max_q = config.qwen.max_question_tokens
    max_a = config.qwen.max_answer_tokens

    def _build(batch_size: int) -> dict:
        return {
            "doc_input_ids": torch.randint(
                0, vocab_size, (batch_size, max_doc), device=device),
            "doc_attention_mask": torch.ones(
                batch_size, max_doc, dtype=torch.long, device=device),
            "q_input_ids": torch.randint(
                0, vocab_size, (batch_size, max_q), device=device),
            "q_attention_mask": torch.ones(
                batch_size, max_q, dtype=torch.long, device=device),
            "answer_ids": torch.randint(
                0, vocab_size, (batch_size, max_a), device=device),
            "answer_attention_mask": torch.ones(
                batch_size, max_a, dtype=torch.long, device=device),
            "answer_labels": torch.randint(
                0, vocab_size, (batch_size, max_a), device=device),
        }

    return _build


def build_sample_batch_fn_lora(
    max_seq_len: int,
    vocab_size: int,
    pad_token_id: int,
    device: torch.device,
):
    """Return a closure that builds synthetic LoRA batches for memory probing."""

    def _build(batch_size: int) -> dict:
        return {
            "input_ids": torch.randint(
                0, vocab_size, (batch_size, max_seq_len), device=device),
            "attention_mask": torch.ones(
                batch_size, max_seq_len, dtype=torch.long, device=device),
            "labels": torch.randint(
                0, vocab_size, (batch_size, max_seq_len), device=device),
        }

    return _build
