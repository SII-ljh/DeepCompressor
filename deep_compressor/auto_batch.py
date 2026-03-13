"""Auto batch size detection and gradient accumulation computation."""

from __future__ import annotations

import gc
import logging
import math
import warnings
from typing import Callable

import torch

logger = logging.getLogger(__name__)


def _is_oom(e: Exception) -> bool:
    """Check if an exception is an OOM error."""
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    if isinstance(e, RuntimeError) and "out of memory" in str(e).lower():
        return True
    return False


def _get_gpu_utilization() -> tuple[float, int, int]:
    """Return (utilization_ratio, used_MB, total_MB) for current CUDA device."""
    if not torch.cuda.is_available():
        return 0.0, 0, 0
    allocated = torch.cuda.memory_allocated()
    reserved = torch.cuda.memory_reserved()
    total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory
    # Use max(allocated, reserved) as the effective usage
    used = max(allocated, reserved)
    return used / total, used // (1 << 20), total // (1 << 20)


def find_max_batch_size(
    model: torch.nn.Module,
    build_batch_fn: Callable[[int], dict],
    forward_fn: Callable[[torch.nn.Module, dict], torch.Tensor],
    device: torch.device,
    max_batch: int = 64,
    min_batch: int = 1,
    mixed_precision: str = "no",
) -> int:
    """Find maximum batch size that achieves high GPU memory utilization.

    Two-phase search:
      1. Descending power-of-2 to find the rough range [success, fail).
      2. Binary search refinement between success and fail to find the
         true maximum batch size, targeting >90% GPU memory utilization.

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
        Maximum batch size that fits in GPU memory.
    """
    if device.type not in ("cuda",):
        warnings.warn(
            f"Auto batch size detection not supported on {device.type}. "
            f"Returning conservative default of 4. "
            f"Set batch_size explicitly for best performance.",
            stacklevel=2,
        )
        return 4

    amp_dtype = None
    if mixed_precision == "fp16":
        amp_dtype = torch.float16
    elif mixed_precision == "bf16":
        amp_dtype = torch.bfloat16

    def _try_batch(batch_size: int) -> bool:
        """Try a forward+backward pass with the given batch size.

        Returns True if it fits, False on OOM.
        Raises non-OOM errors.
        """
        try:
            _cleanup_gpu()
            batch = build_batch_fn(batch_size)

            if amp_dtype is not None:
                with torch.autocast(device_type="cuda", dtype=amp_dtype):
                    loss = forward_fn(model, batch)
            else:
                loss = forward_fn(model, batch)

            loss.backward()

            model.zero_grad(set_to_none=True)
            del loss, batch
            _cleanup_gpu()
            return True

        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if not _is_oom(e):
                raise
            model.zero_grad(set_to_none=True)
            _cleanup_gpu()
            return False

    # ── Phase 1: Descending power-of-2 to find rough range ──
    # Highest power of 2 <= max_batch
    probe = 1
    while probe * 2 <= max_batch:
        probe *= 2

    lo = min_batch  # largest known-good
    hi = None       # smallest known-bad (None = not found yet)

    while probe >= min_batch:
        if _try_batch(probe):
            lo = probe
            logger.info(f"Auto batch size phase 1: {probe} fits")
            break
        else:
            hi = probe
            logger.info(f"Auto batch size phase 1: {probe} OOM")
            probe //= 2

    if lo < min_batch:
        lo = min_batch

    # If the largest power of 2 (== max_batch cap) succeeded and no OOM
    # was seen above it, try max_batch directly as it may be higher.
    if hi is None and lo < max_batch:
        if _try_batch(max_batch):
            lo = max_batch
            logger.info(f"Auto batch size: max_batch={max_batch} fits directly")

    # If no OOM boundary found, set hi = lo + 1 (nothing to refine)
    if hi is None:
        hi = lo + 1

    # ── Phase 2: Binary search refinement between lo and hi ──
    # lo fits, hi OOMs (or is beyond max_batch). Find true max in [lo, hi).
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if mid > max_batch:
            hi = mid
            continue
        if _try_batch(mid):
            lo = mid
            logger.info(f"Auto batch size phase 2: {mid} fits")
        else:
            hi = mid
            logger.info(f"Auto batch size phase 2: {mid} OOM")

    best = lo

    # ── Final: probe once more at best to report memory utilization ──
    if torch.cuda.is_available():
        _try_batch(best)
        util, used_mb, total_mb = _get_gpu_utilization()
        logger.info(
            f"Auto batch size result: {best}  "
            f"(GPU mem: {used_mb}MB / {total_mb}MB = {util:.1%})"
        )
    else:
        logger.info(f"Auto batch size result: {best}")

    return best


def _cleanup_gpu():
    """Release GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def sync_batch_size_across_ranks(local_batch_size: int) -> int:
    """Take the minimum batch size across all DDP ranks.

    In multi-GPU setups, different GPUs may have different available memory
    (mixed GPU types, other processes occupying VRAM, etc.).  Each rank probes
    independently, then we synchronize to the **minimum** so that every rank
    can handle the chosen batch size without OOM.

    If ``torch.distributed`` is not initialized or WORLD_SIZE <= 1, returns
    *local_batch_size* unchanged.
    """
    import os

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return local_batch_size

    import torch.distributed as dist

    # Initialize process group if not already done (accelerate launch sets
    # the env vars but may not have called init_process_group yet).
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        _did_init = True
    else:
        _did_init = False

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    tensor = torch.tensor([local_batch_size], dtype=torch.long, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    global_min = tensor.item()

    if global_min != local_batch_size:
        logger.warning(
            f"Rank {local_rank}: local max batch={local_batch_size}, "
            f"global min={global_min} (constrained by weakest GPU)"
        )
    else:
        logger.info(
            f"Rank {local_rank}: batch size {local_batch_size} "
            f"consistent across all ranks"
        )

    # Don't destroy the process group — Accelerator will reuse it
    return global_min


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
