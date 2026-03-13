"""Training script for Deep Compressor with HuggingFace Accelerate.

Supports:
  - Single GPU / MPS / CPU  (auto-detected)
  - Multi-GPU DDP            via ``accelerate launch``
  - Mixed precision fp16/bf16
  - Gradient checkpointing   (config flag)
  - wandb experiment tracking (via Accelerate TrackerMixin)
  - Hydra configuration management (optional)
  - Optuna hyperparameter search callback

Usage:
  python -m deep_compressor.train --config configs/default.yaml \\
      --data_path data/qa_train.json --eval_data_path data/qa_dev.json

  # With wandb
  python -m deep_compressor.train --config configs/default.yaml \\
      --data_path data/qa_train.json --eval_data_path data/qa_dev.json \\
      --wandb --wandb_project deep-compressor

  # Multi-GPU
  accelerate launch --multi_gpu --num_processes 4 \\
      -m deep_compressor.train --config configs/default.yaml \\
      --data_path data/qa_train.json --eval_data_path data/qa_dev.json
"""

import argparse
import gc
import logging
import math
import os
import re
import sys
from datetime import datetime

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from deep_compressor.config import DeepCompressorConfig
from deep_compressor.data import PaddingCollator, QADataset
from deep_compressor.eval import evaluate_qa
from deep_compressor.model import DeepCompressor

logger = logging.getLogger(__name__)


# ── wandb naming helpers ──────────────────────────────────────────────
def _short_model_name(model_name_or_path: str) -> str:
    """Extract short model identifier, e.g. 'Qwen/Qwen3-0.6B' -> 'qwen3-0.6B'."""
    basename = model_name_or_path.rstrip("/").split("/")[-1]
    # Remove common prefixes, keep concise
    basename = re.sub(r"^[Qq]wen-?", "", basename)
    return basename.lower() if basename else "unknown"


def generate_run_name(config: DeepCompressorConfig,
                      timestamp: str | None = None) -> str:
    """Auto-generate a descriptive wandb run name from config.

    Format: q{num_queries}_lr{lr}_{model}_{timestamp}
    Example: q64_lr1e-4_qwen3-0.6b_20260311_143052
    """
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    q = config.effective_num_queries
    lr = config.training.learning_rate
    model = _short_model_name(config.qwen.model_name_or_path)

    # Format lr compactly: 1e-4, 5e-5, etc.
    lr_str = f"{lr:.0e}".replace("+", "")

    return f"q{q}_lr{lr_str}_{model}_{timestamp}"


def generate_auto_tags(config: DeepCompressorConfig) -> list[str]:
    """Auto-generate wandb tags from config for filtering/grouping."""
    tags = []

    # Model
    model = _short_model_name(config.qwen.model_name_or_path)
    tags.append(f"model:{model}")

    # Compression ratio
    q = config.effective_num_queries
    tags.append(f"q{q}")

    # Document length
    tags.append(f"doc{config.qwen.max_doc_tokens}")

    # Projection modes (only if non-default)
    if config.ablation.down_proj_mode != "identity":
        tags.append(f"down:{config.ablation.down_proj_mode}")
    if config.ablation.up_proj_mode != "identity":
        tags.append(f"up:{config.ablation.up_proj_mode}")

    # Perceiver stage ablations
    if not config.ablation.enable_stage_a:
        tags.append("no-stage-a")
    if not config.ablation.enable_stage_b:
        tags.append("no-stage-b")
    if not config.ablation.enable_stage_c:
        tags.append("no-stage-c")

    # Key hyperparameters
    lr = config.training.learning_rate
    tags.append(f"lr:{lr:.0e}".replace("+", ""))
    bs = config.training.batch_size * config.training.gradient_accumulation_steps
    tags.append(f"ebs{bs}")

    return tags


def build_wandb_conf(args) -> "SimpleNamespace | None":
    """Build wandb SimpleNamespace config from argparse args + DeepCompressorConfig.

    Auto-generates run_name and tags when not explicitly provided.
    """
    if not args.wandb:
        return None

    from types import SimpleNamespace

    config = DeepCompressorConfig.from_yaml(args.config)

    # Auto-generate run name if not provided
    run_name = args.wandb_run_name
    if not run_name:
        run_name = generate_run_name(config)

    # Merge explicit tags with auto-generated ones
    explicit_tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()] \
        if args.wandb_tags else []
    auto_tags = generate_auto_tags(config)
    # Explicit tags first, then auto tags (dedup preserving order)
    seen = set()
    merged_tags = []
    for t in explicit_tags + auto_tags:
        if t not in seen:
            seen.add(t)
            merged_tags.append(t)

    return SimpleNamespace(
        enabled=True,
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=run_name,
        tags=merged_tags,
        group=args.wandb_group,
        notes=args.wandb_notes,
        offline=args.wandb_offline,
    )


# ── helpers ────────────────────────────────────────────────────────────
def get_scheduler(optimizer, warmup_steps: int, total_steps: int,
                  scheduler_type: str):
    if scheduler_type == "cosine":
        warmup = LinearLR(optimizer, start_factor=1e-8 / optimizer.defaults["lr"],
                          end_factor=1.0, total_iters=warmup_steps)
        cosine = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps)
        return SequentialLR(optimizer, [warmup, cosine],
                            milestones=[warmup_steps])
    return LinearLR(optimizer, start_factor=1e-8 / optimizer.defaults["lr"],
                    end_factor=1.0, total_iters=warmup_steps)


def save_checkpoint(model, accelerator: Accelerator, output_dir: str, tag):
    """Save trainable weights only (skip frozen Qwen)."""
    unwrapped = accelerator.unwrap_model(model)
    save_path = os.path.join(output_dir, f"checkpoint-{tag}")
    os.makedirs(save_path, exist_ok=True)
    state = {k: v.cpu() for k, v in unwrapped.state_dict().items()
             if not k.startswith("qwen.")}
    torch.save(state, os.path.join(save_path, "trainable_weights.pt"))
    logger.info(f"Saved checkpoint to {save_path}")


def _build_forward_kwargs(batch, completed_steps: int):
    """Build keyword arguments for model.forward()."""
    kwargs = dict(
        doc_input_ids=batch["doc_input_ids"],
        doc_attention_mask=batch["doc_attention_mask"],
        q_input_ids=batch["q_input_ids"],
        q_attention_mask=batch["q_attention_mask"],
        answer_ids=batch["answer_ids"],
        answer_attention_mask=batch["answer_attention_mask"],
        answer_labels=batch["answer_labels"],
        global_step=completed_steps,
    )
    # Teacher outputs (optional, for distillation)
    if "teacher_logits" in batch:
        kwargs["teacher_logits"] = batch["teacher_logits"]
    if "teacher_hidden" in batch:
        kwargs["teacher_hidden"] = batch["teacher_hidden"]
    return kwargs


def _split_batch(batch: dict, chunks: int) -> list:
    """Split a batch dict along dim=0 into *chunks* sub-batches."""
    bs = next(iter(batch.values())).shape[0]
    indices = [range(i, min(i + (bs + chunks - 1) // chunks, bs))
               for i in range(0, bs, (bs + chunks - 1) // chunks)]
    sub_batches = []
    for idx in indices:
        idx = list(idx)
        if len(idx) == 0:
            continue
        sub = {k: v[idx] for k, v in batch.items()}
        sub_batches.append(sub)
    return sub_batches


def _run_batch_with_oom_retry(
    batch, model, teacher_model, accelerator, optimizer,
    scheduler, tcfg, completed_steps, build_fwd_fn,
    _max_splits: int = 3,
):
    """Run a training step, splitting the batch on OOM instead of skipping.

    On OOM, the batch is split into 2x smaller sub-batches and retried.
    Gradients are accumulated across sub-batches so all data participates
    in training.  Up to *_max_splits* halvings (batch/2, batch/4, batch/8).

    Returns:
        dict of averaged loss components, or None if all retries failed.
    """
    bs = next(iter(batch.values())).shape[0]

    def _forward_one(sub_batch, num_sub_batches: int):
        """Forward + backward on one (sub-)batch, scaling loss."""
        fwd_kwargs = build_fwd_fn(sub_batch, completed_steps)

        if teacher_model is not None:
            with torch.no_grad():
                t_input_ids = torch.cat([
                    sub_batch["doc_input_ids"],
                    sub_batch["q_input_ids"],
                    sub_batch["answer_ids"],
                ], dim=1)
                t_attention_mask = torch.cat([
                    sub_batch["doc_attention_mask"],
                    sub_batch["q_attention_mask"],
                    sub_batch["answer_attention_mask"],
                ], dim=1)
                t_out = teacher_model(
                    input_ids=t_input_ids,
                    attention_mask=t_attention_mask,
                    output_hidden_states=True, use_cache=False,
                )
                doc_len = sub_batch["doc_input_ids"].shape[1]
                fwd_kwargs["teacher_logits"] = \
                    t_out.logits[:, doc_len:, :].detach()
                fwd_kwargs["teacher_hidden"] = [
                    h[:, doc_len:, :].detach()
                    for h in t_out.hidden_states]

        losses = model(**fwd_kwargs)

        # Scale loss by 1/num_sub_batches so gradient accumulation across
        # sub-batches equals the original full-batch gradient.
        scaled_loss = losses["total"] / num_sub_batches
        accelerator.backward(scaled_loss)

        return {k: (v.detach().item() if torch.is_tensor(v) else v)
                for k, v in losses.items()}

    # Try full batch first, split on OOM
    num_chunks = 1
    for attempt in range(_max_splits + 1):
        try:
            if num_chunks == 1:
                # Full batch — normal path with accelerator.accumulate
                with accelerator.accumulate(model):
                    fwd_kwargs = build_fwd_fn(batch, completed_steps)

                    if teacher_model is not None:
                        with torch.no_grad():
                            t_input_ids = torch.cat([
                                batch["doc_input_ids"],
                                batch["q_input_ids"],
                                batch["answer_ids"],
                            ], dim=1)
                            t_attention_mask = torch.cat([
                                batch["doc_attention_mask"],
                                batch["q_attention_mask"],
                                batch["answer_attention_mask"],
                            ], dim=1)
                            t_out = teacher_model(
                                input_ids=t_input_ids,
                                attention_mask=t_attention_mask,
                                output_hidden_states=True, use_cache=False,
                            )
                            doc_len = batch["doc_input_ids"].shape[1]
                            fwd_kwargs["teacher_logits"] = \
                                t_out.logits[:, doc_len:, :].detach()
                            fwd_kwargs["teacher_hidden"] = [
                                h[:, doc_len:, :].detach()
                                for h in t_out.hidden_states]

                    losses = model(**fwd_kwargs)
                    accelerator.backward(losses["total"])

                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            model.parameters(), tcfg.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                return {k: (v.detach().item() if torch.is_tensor(v) else v)
                        for k, v in losses.items()}
            else:
                # Split path — manual gradient accumulation across sub-batches
                sub_batches = _split_batch(batch, num_chunks)
                all_losses = []
                for sub in sub_batches:
                    sub_losses = _forward_one(sub, len(sub_batches))
                    all_losses.append(sub_losses)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        model.parameters(), tcfg.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # Average loss components across sub-batches
                avg = {}
                for k in all_losses[0]:
                    avg[k] = sum(sl[k] for sl in all_losses) / len(all_losses)
                return avg

        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            # OOM — clean up and retry with more splits
            model.zero_grad(set_to_none=True)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            num_chunks *= 2
            if attempt < _max_splits:
                logger.warning(
                    f"Step {completed_steps}: OOM on batch (bs={bs}, "
                    f"doc={batch['doc_input_ids'].shape[1]}), "
                    f"retrying with {num_chunks} sub-batches")
            else:
                logger.error(
                    f"Step {completed_steps}: OOM persists after "
                    f"{_max_splits} splits (bs=1), skipping batch")
                return None

    return None


# ── unified training loop ──────────────────────────────────────────────
def train_stage(config: DeepCompressorConfig, model: DeepCompressor,
                train_loader: DataLoader, accelerator: Accelerator,
                eval_loader: DataLoader = None,
                tokenizer=None, optuna_callback=None,
                diagnostic_callback=None, teacher_model=None):
    """Train QA model.

    Args:
        eval_loader: optional DataLoader for validation
        tokenizer: required for QA eval (decoding generated tokens)
        optuna_callback: optional callable(step, metrics) for Optuna pruning
        diagnostic_callback: optional callable(step, model, accelerator) -> dict
        teacher_model: optional frozen Qwen model for on-the-fly distillation.
                       When provided, teacher logits/hidden states are computed
                       per batch rather than pre-cached.
    """
    tcfg = config.training

    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay,
    )
    scheduler = get_scheduler(optimizer, tcfg.warmup_steps,
                              tcfg.max_steps, tcfg.scheduler)

    # accelerate.prepare handles DDP wrapping, device placement, etc.
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler,
    )
    if eval_loader is not None:
        eval_loader = accelerator.prepare(eval_loader)

    model.train()
    completed_steps = 0
    running_loss = 0.0
    running_components = {}  # accumulate per-component losses
    micro_steps = 0
    last_metrics = None
    best_metric = float("inf")  # track best eval loss for saving best checkpoint
    patience_counter = 0  # early stopping counter

    current_epoch = 0

    while completed_steps < tcfg.max_steps:
        current_epoch += 1
        if tcfg.epochs > 0 and accelerator.is_main_process:
            logger.info(f"Epoch {current_epoch}/{tcfg.epochs}")
        for batch in train_loader:
            # _run_batch_with_oom_retry processes the batch, splitting on OOM
            batch_losses = _run_batch_with_oom_retry(
                batch, model, teacher_model, accelerator, optimizer,
                scheduler, tcfg, completed_steps, _build_forward_kwargs,
            )
            if batch_losses is None:
                # All sub-batch retries failed — extremely rare
                continue

            running_loss += batch_losses["total"]
            micro_steps += 1

            # accumulate loss components
            for k, v in batch_losses.items():
                if k == "total":
                    continue
                running_components[k] = running_components.get(k, 0.0) + v

            if accelerator.sync_gradients:
                completed_steps += 1

                # ── logging ──
                if completed_steps % tcfg.log_every == 0:
                    avg_local = running_loss / max(micro_steps, 1)
                    avg_tensor = torch.tensor([avg_local],
                                              device=accelerator.device)
                    avg_loss = accelerator.gather(avg_tensor).mean().item()
                    if accelerator.is_main_process:
                        lr = scheduler.get_last_lr()[0]
                        ppl = torch.exp(torch.tensor(avg_loss)).item()
                        logger.info(
                            f"[QA] step {completed_steps}/{tcfg.max_steps}  "
                            f"loss={avg_loss:.4f}  ppl={ppl:.2f}  lr={lr:.2e}")

                        # log to wandb via accelerator
                        log_dict = {
                            "qa/loss": avg_loss,
                            "qa/ppl": ppl,
                            "qa/lr": lr,
                        }
                        for k, v in running_components.items():
                            log_dict[f"qa/{k}"] = v / max(micro_steps, 1)
                        accelerator.log(log_dict, step=completed_steps)

                    running_loss = 0.0
                    running_components = {}
                    micro_steps = 0

                # ── checkpoint ──
                if completed_steps % tcfg.save_every == 0:
                    accelerator.wait_for_everyone()
                    if accelerator.is_main_process:
                        save_checkpoint(model, accelerator,
                                        tcfg.output_dir, completed_steps)

                # ── evaluation ──
                if (eval_loader is not None
                        and completed_steps % tcfg.eval_every == 0):
                    accelerator.wait_for_everyone()
                    model.eval()
                    metrics = evaluate_qa(
                        model, eval_loader, tokenizer, accelerator,
                        max_new_tokens=config.qwen.max_answer_tokens,
                        show_samples=5)
                    if accelerator.is_main_process:
                        logger.info(
                            f"[QA EVAL] step {completed_steps}  "
                            f"loss={metrics['loss']:.4f}  "
                            f"ppl={metrics['perplexity']:.2f}  "
                            f"EM={metrics['exact_match']:.2%}  "
                            f"F1={metrics['f1']:.4f}")

                    # log eval metrics
                    if accelerator.is_main_process:
                        accelerator.log(
                            {f"eval/{k}": v for k, v in metrics.items()},
                            step=completed_steps,
                        )

                    last_metrics = metrics

                    # Save best checkpoint (by eval loss)
                    eval_loss = metrics.get("loss", float("inf"))
                    if eval_loss < best_metric:
                        best_metric = eval_loss
                        patience_counter = 0
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            save_checkpoint(model, accelerator,
                                            tcfg.output_dir, "best")
                            logger.info(
                                f"New best eval loss: {eval_loss:.4f} "
                                f"at step {completed_steps}")
                    else:
                        patience_counter += 1
                        if accelerator.is_main_process:
                            logger.info(
                                f"No improvement for {patience_counter} eval(s) "
                                f"(best={best_metric:.4f}, current={eval_loss:.4f})")

                    # Early stopping
                    if (tcfg.early_stopping_patience > 0
                            and patience_counter >= tcfg.early_stopping_patience):
                        if accelerator.is_main_process:
                            logger.info(
                                f"Early stopping triggered at step {completed_steps} "
                                f"(patience={tcfg.early_stopping_patience})")
                        return model, last_metrics

                    model.train()

                    # Diagnostic callback (mid-training diagnostics)
                    if diagnostic_callback is not None:
                        try:
                            diag_results = diagnostic_callback(
                                completed_steps, model, accelerator)
                            if diag_results and accelerator.is_main_process:
                                accelerator.log(
                                    {f"diagnostic/{k}": v
                                     for k, v in diag_results.items()
                                     if isinstance(v, (int, float))},
                                    step=completed_steps,
                                )
                        except Exception as e:
                            logger.warning(f"Diagnostic callback failed: {e}")

                    # Optuna pruning callback
                    if optuna_callback is not None:
                        optuna_callback(completed_steps, metrics)

                if completed_steps >= tcfg.max_steps:
                    break

    return model, last_metrics


# ── core training entrypoint ──────────────────────────────────────────
def _run_training(config: DeepCompressorConfig,
                  data_path: str,
                  eval_data_path: str = None,
                  resume_from: str = None,
                  max_eval_samples: int = 0,
                  max_train_samples: int = 0,
                  wandb_conf=None,
                  optuna_callback=None,
                  diagnostic_callback=None):
    """Core training function shared by legacy argparse and Hydra entry points.

    Args:
        config: fully-built DeepCompressorConfig
        data_path: path to training data
        eval_data_path: optional path to eval data (QA dev set or NTP val)
        resume_from: optional checkpoint directory to resume from
        max_eval_samples: limit eval samples (0 = all)
        max_train_samples: limit training samples (0 = all)
        wandb_conf: optional wandb configuration (object with .enabled, .project, etc.)
        optuna_callback: optional callable(step, metrics) for Optuna pruning
        diagnostic_callback: optional callable(step, model, accelerator) -> dict

    Returns:
        dict or None: last evaluation metrics (useful for Optuna)
    """
    tcfg = config.training

    # ── early logging (before Accelerator) ──
    _is_main = os.environ.get("RANK", "0") == "0"
    if _is_main:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s - %(name)s - %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

    # ── tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(
        config.qwen.model_name_or_path,
        trust_remote_code=True,
        fix_mistral_regex=True  # Fix tokenizer regex warning
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── model ──
    model = DeepCompressor(config)

    if tcfg.gradient_checkpointing:
        model.qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        logger.info("Gradient checkpointing enabled for Qwen")

    if resume_from:
        weights = torch.load(
            os.path.join(resume_from, "trainable_weights.pt"),
            map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=False)
        logger.info(f"Resumed from {resume_from}")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Trainable: {trainable:,} / {total:,} "
                f"({100*trainable/total:.1f}%)")

    # ── auto batch size detection (before Accelerator) ──
    if tcfg.auto_batch_size:
        from deep_compressor.auto_batch import (
            build_probe_fn_from_dataset,
            build_sample_batch_fn_qa,
            compute_gradient_accumulation,
            find_max_batch_size,
            sync_batch_size_across_ranks,
        )

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            probe_device = torch.device(f"cuda:{local_rank}")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            probe_device = torch.device("mps")
        else:
            probe_device = torch.device("cpu")

        model.to(probe_device)

        # Build probe batches from real data (P95 lengths) when possible,
        # fall back to synthetic worst-case otherwise.
        _probe_dataset = None
        try:
            _probe_dataset = QADataset(
                data_path, tokenizer,
                max_doc_tokens=config.qwen.max_doc_tokens,
                max_question_tokens=config.qwen.max_question_tokens,
                max_answer_tokens=config.qwen.max_answer_tokens)
        except Exception as e:
            logger.warning(f"Could not load dataset for probe: {e}")

        if _probe_dataset is not None and len(_probe_dataset) > 0:
            _collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
            build_fn, data_stats = build_probe_fn_from_dataset(
                _probe_dataset, _collator, probe_device,
                percentile=95.0, num_stat_samples=500)
            del _probe_dataset, _collator
        else:
            logger.warning("Falling back to synthetic worst-case probe")
            build_fn = build_sample_batch_fn_qa(
                config, config.qwen.vocab_size, probe_device)

        def _probe_forward(m, batch):
            losses = m(**batch, global_step=0)
            return losses["total"]

        max_bs = find_max_batch_size(
            model, build_fn, _probe_forward, probe_device,
            mixed_precision=tcfg.mixed_precision,
        )

        # Sync across ranks — take the minimum so every GPU can handle it
        max_bs = sync_batch_size_across_ranks(max_bs)
        tcfg.batch_size = max_bs

        num_gpus = int(os.environ.get("WORLD_SIZE", "1"))
        tcfg.gradient_accumulation_steps = compute_gradient_accumulation(
            max_bs, num_gpus, tcfg.target_effective_batch_size)

        effective = max_bs * num_gpus * tcfg.gradient_accumulation_steps
        logger.info(
            f"Auto batch: bs={max_bs}, grad_accum={tcfg.gradient_accumulation_steps}, "
            f"num_gpus={num_gpus}, effective={effective} "
            f"(target={tcfg.target_effective_batch_size})")

        # Move model back to CPU for Accelerator to handle device placement
        model.to("cpu")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── accelerator ──
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)

    log_with = []
    if wandb_conf and getattr(wandb_conf, "enabled", False):
        log_with.append("wandb")

    accelerator = Accelerator(
        gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
        mixed_precision=tcfg.mixed_precision,
        kwargs_handlers=[ddp_kwargs],
        log_with=log_with or None,
    )

    logger.info(f"Device: {accelerator.device}  |  "
                f"Processes: {accelerator.num_processes}  |  "
                f"Mixed precision: {accelerator.mixed_precision}")

    # ── init wandb tracker ──
    if log_with:
        init_kwargs = {}
        if wandb_conf:
            wb_init = {}
            if getattr(wandb_conf, "entity", None):
                wb_init["entity"] = wandb_conf.entity
            if getattr(wandb_conf, "run_name", None):
                wb_init["name"] = wandb_conf.run_name
            if getattr(wandb_conf, "tags", None):
                wb_init["tags"] = list(wandb_conf.tags)
            if getattr(wandb_conf, "group", None):
                wb_init["group"] = wandb_conf.group
            if getattr(wandb_conf, "notes", None):
                wb_init["notes"] = wandb_conf.notes
            if getattr(wandb_conf, "offline", False):
                os.environ["WANDB_MODE"] = "offline"
            init_kwargs["wandb"] = wb_init

        # Flat hyperparameters for easy wandb dashboard filtering
        flat_config = config.to_dict()
        flat_config["_hp/num_queries"] = config.effective_num_queries
        flat_config["_hp/learning_rate"] = config.training.learning_rate
        flat_config["_hp/batch_size"] = config.training.batch_size
        flat_config["_hp/effective_batch_size"] = (
            config.training.batch_size
            * config.training.gradient_accumulation_steps
        )
        flat_config["_hp/max_doc_tokens"] = config.qwen.max_doc_tokens
        flat_config["_hp/max_steps"] = config.training.max_steps
        flat_config["_hp/model"] = config.qwen.model_name_or_path
        flat_config["_hp/down_proj_mode"] = config.ablation.down_proj_mode
        flat_config["_hp/up_proj_mode"] = config.ablation.up_proj_mode
        flat_config["_hp/mixed_precision"] = config.training.mixed_precision

        accelerator.init_trackers(
            project_name=getattr(wandb_conf, "project", "deep-compressor"),
            config=flat_config,
            init_kwargs=init_kwargs,
        )

    # ── dataloader ──
    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
    # MPS doesn't support multi-process data loading reliably
    num_workers = 0 if accelerator.device.type == "mps" else 2
    pin_memory = accelerator.device.type == "cuda"

    os.makedirs(tcfg.output_dir, exist_ok=True)

    dataset = QADataset(data_path, tokenizer,
                        max_doc_tokens=config.qwen.max_doc_tokens,
                        max_question_tokens=config.qwen.max_question_tokens,
                        max_answer_tokens=config.qwen.max_answer_tokens)

    # Truncate training data if requested
    if max_train_samples > 0 and max_train_samples < len(dataset):
        dataset = Subset(dataset,
                         list(range(max_train_samples)))
        logger.info(f"Truncated QA dataset to {max_train_samples} samples")

    loader = DataLoader(dataset, batch_size=tcfg.batch_size, shuffle=True,
                        collate_fn=collator, num_workers=num_workers,
                        pin_memory=pin_memory)
    logger.info(f"QA dataset: {len(dataset):,} samples")

    eval_loader = None
    if eval_data_path:
        eval_ds = QADataset(eval_data_path, tokenizer,
                            max_doc_tokens=config.qwen.max_doc_tokens,
                            max_question_tokens=config.qwen.max_question_tokens,
                            max_answer_tokens=config.qwen.max_answer_tokens)
        if max_eval_samples > 0:
            eval_ds = Subset(eval_ds,
                             list(range(min(max_eval_samples,
                                            len(eval_ds)))))
        eval_loader = DataLoader(eval_ds, batch_size=tcfg.batch_size,
                                 shuffle=False, collate_fn=collator,
                                 num_workers=num_workers,
                                 pin_memory=pin_memory)
        logger.info(f"QA eval: {len(eval_ds):,} samples")

    # ── epoch-based training ──
    if tcfg.epochs > 0:
        dataset_size = len(dataset)
        effective_batch = (tcfg.batch_size
                           * tcfg.gradient_accumulation_steps
                           * accelerator.num_processes)
        steps_per_epoch = math.ceil(dataset_size / effective_batch)
        tcfg.max_steps = steps_per_epoch * tcfg.epochs

        # Auto-set warmup to 5% if still at default (1000)
        if tcfg.warmup_steps == 1000:
            tcfg.warmup_steps = max(1, int(0.05 * tcfg.max_steps))

        logger.info(
            f"Epoch-based training: {tcfg.epochs} epochs, "
            f"{steps_per_epoch} steps/epoch, "
            f"{tcfg.max_steps} total steps, "
            f"warmup={tcfg.warmup_steps}")

    last_metrics = None
    _, last_metrics = train_stage(
        config, model, loader, accelerator,
        eval_loader=eval_loader, tokenizer=tokenizer,
        optuna_callback=optuna_callback,
        diagnostic_callback=diagnostic_callback)

    # ── final save ──
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_checkpoint(model, accelerator, tcfg.output_dir, "final")

    accelerator.end_training()
    logger.info("Training complete.")

    return last_metrics


# ── legacy argparse entry ─────────────────────────────────────────────
def main_legacy():
    """Backward-compatible argparse entry point."""
    parser = argparse.ArgumentParser(description="Train Deep Compressor")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint dir")
    parser.add_argument("--eval_data_path", type=str, default=None,
                        help="Path to QA dev set")
    parser.add_argument("--max_eval_samples", type=int, default=0,
                        help="Limit eval samples (0 = all)")
    parser.add_argument("--max_train_samples", type=int, default=0,
                        help="Limit training samples (0 = all)")
    # wandb options
    parser.add_argument("--wandb", action="store_true",
                        help="Enable wandb logging")
    parser.add_argument("--wandb_project", type=str, default="deep-compressor",
                        help="wandb project name")
    parser.add_argument("--wandb_entity", type=str, default=None,
                        help="wandb entity (team/user)")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="wandb run name (auto-generated if not set)")
    parser.add_argument("--wandb_tags", type=str, default=None,
                        help="Comma-separated wandb tags (merged with auto tags)")
    parser.add_argument("--wandb_group", type=str, default=None,
                        help="wandb group for grouping related runs")
    parser.add_argument("--wandb_notes", type=str, default=None,
                        help="wandb run notes/description")
    parser.add_argument("--wandb_offline", action="store_true",
                        help="Use wandb in offline mode")
    # Diagnostic callback options
    parser.add_argument("--diagnostic_every", type=int, default=0,
                        help="Run mid-training diagnostics every N steps (0 = off)")
    parser.add_argument("--diagnostic_data_path", type=str, default=None,
                        help="Data path for diagnostic callback (defaults to --data_path)")
    parser.add_argument("--diagnostic_experiments", type=str, default="4,5",
                        help="Comma-separated mid-training experiments to run")
    # Auto batch / epoch overrides
    parser.add_argument("--auto_batch_size", action="store_true",
                        help="Auto-detect max per-GPU batch size (overrides config)")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (overrides config; 0=use max_steps)")
    parser.add_argument("--target_effective_batch_size", type=int, default=None,
                        help="Target effective batch size for auto grad_accum (default: 256)")
    args = parser.parse_args()

    config = DeepCompressorConfig.from_yaml(args.config)

    # CLI overrides for training config
    if args.auto_batch_size:
        config.training.auto_batch_size = True
    if args.epochs is not None:
        config.training.epochs = args.epochs
    if args.target_effective_batch_size is not None:
        config.training.target_effective_batch_size = args.target_effective_batch_size

    # Build wandb config from CLI flags (auto-generates name/tags if not set)
    wandb_conf = build_wandb_conf(args)

    # Build diagnostic callback if requested
    diag_callback = None
    if args.diagnostic_every > 0:
        from scripts.diagnostics.mid_training import create_mid_training_callback
        diag_data = args.diagnostic_data_path or args.data_path
        diag_experiments = tuple(
            e.strip() for e in args.diagnostic_experiments.split(","))
        diag_callback = create_mid_training_callback(
            config, diag_data, experiments=diag_experiments,
            run_every=args.diagnostic_every,
        )

    _run_training(
        config=config,
        data_path=args.data_path,
        eval_data_path=args.eval_data_path,
        resume_from=args.resume_from,
        max_eval_samples=args.max_eval_samples,
        max_train_samples=args.max_train_samples,
        wandb_conf=wandb_conf,
        diagnostic_callback=diag_callback,
    )


# ── Hydra entry ───────────────────────────────────────────────────────
def main_hydra():
    """Hydra-based entry point with structured config."""
    import hydra
    from omegaconf import DictConfig, OmegaConf

    # Import to trigger ConfigStore registration
    from deep_compressor.hydra_conf import RunConf, WandbConf, to_deep_compressor_config  # noqa: F401

    @hydra.main(version_base=None, config_path=None, config_name="base_config")
    def _hydra_main(cfg: DictConfig) -> None:
        config = to_deep_compressor_config(cfg)

        # Resolve relative paths against original cwd (Hydra changes cwd)
        orig_cwd = hydra.utils.get_original_cwd()

        def _resolve_path(p):
            if p and not os.path.isabs(p):
                return os.path.join(orig_cwd, p)
            return p

        data_path = _resolve_path(cfg.data_path)
        eval_data_path = _resolve_path(cfg.get("eval_data_path"))
        resume_from = _resolve_path(cfg.get("resume_from"))

        # Resolve output_dir too
        if not os.path.isabs(config.training.output_dir):
            config.training.output_dir = os.path.join(
                orig_cwd, config.training.output_dir)

        # Build wandb conf
        wandb_conf = None
        if "wandb" in cfg:
            from types import SimpleNamespace
            wb = OmegaConf.to_container(cfg.wandb, resolve=True)
            wandb_conf = SimpleNamespace(**wb)

        _run_training(
            config=config,
            data_path=data_path,
            eval_data_path=eval_data_path,
            resume_from=resume_from,
            max_eval_samples=cfg.get("max_eval_samples", 0),
            max_train_samples=cfg.get("max_train_samples", 0),
            wandb_conf=wandb_conf,
        )

    _hydra_main()


# ── dispatcher ────────────────────────────────────────────────────────
def _is_hydra_mode() -> bool:
    """Detect Hydra mode by checking for Hydra-style CLI arguments."""
    for arg in sys.argv[1:]:
        if arg.startswith("--config-path") or arg.startswith("--config-name"):
            return True
        # Hydra overrides: key=value without leading --
        if "=" in arg and not arg.startswith("-"):
            return True
    return False


def main():
    """Dispatch to legacy argparse or Hydra entry based on CLI arguments."""
    if _is_hydra_mode():
        main_hydra()
    else:
        main_legacy()


if __name__ == "__main__":
    main()
