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
  # ── Legacy mode (backward compatible) ──
  python -m deep_compressor.train --config configs/default.yaml \\
      --data_path data/ntp_train.jsonl --stage 1

  # ── Legacy + wandb ──
  python -m deep_compressor.train --config configs/default.yaml \\
      --data_path data/ntp_train.jsonl --stage 1 \\
      --wandb --wandb_project deep-compressor-test

  # ── Hydra mode (CLI overrides) ──
  python -m deep_compressor.train \\
      --config-path ../configs --config-name default \\
      data_path=data/ntp_train.jsonl training.learning_rate=1e-3

  # Multi-GPU
  accelerate launch --multi_gpu --num_processes 4 \\
      -m deep_compressor.train --config configs/default.yaml \\
      --data_path data/ntp_train.jsonl --stage 1
"""

import argparse
import logging
import os
import sys

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from deep_compressor.config import DeepCompressorConfig
from deep_compressor.data import NTPDataset, PaddingCollator, QADataset
from deep_compressor.eval import evaluate_ntp, evaluate_qa
from deep_compressor.model import DeepCompressor

logger = logging.getLogger(__name__)


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


def _build_forward_kwargs(batch, mode: str, completed_steps: int):
    """Build keyword arguments for model.forward() based on training mode."""
    if mode == "ntp":
        return dict(
            mode="ntp",
            doc_input_ids=batch["doc_input_ids"],
            doc_attention_mask=batch["doc_attention_mask"],
            segment_ids=batch["segment_ids"],
            segment_attention_mask=batch["segment_attention_mask"],
            segment_labels=batch["segment_labels"],
        )
    # QA mode
    kwargs = dict(
        mode="qa",
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


# ── unified training loop ──────────────────────────────────────────────
def train_stage(config: DeepCompressorConfig, model: DeepCompressor,
                train_loader: DataLoader, accelerator: Accelerator,
                mode: str, eval_loader: DataLoader = None,
                tokenizer=None, optuna_callback=None,
                diagnostic_callback=None, teacher_model=None):
    """Train one stage (NTP or QA).

    Args:
        mode: "ntp" for Stage 1, "qa" for Stage 2
        eval_loader: optional DataLoader for validation
        tokenizer: required for QA eval (decoding generated tokens)
        optuna_callback: optional callable(step, metrics) for Optuna pruning
        diagnostic_callback: optional callable(step, model, accelerator) -> dict
        teacher_model: optional frozen Qwen model for on-the-fly distillation
                       (QA mode only). When provided, teacher logits/hidden states
                       are computed per batch rather than pre-cached.
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

    while completed_steps < tcfg.max_steps:
        for batch in train_loader:
            with accelerator.accumulate(model):
                fwd_kwargs = _build_forward_kwargs(batch, mode, completed_steps)

                # On-the-fly teacher distillation (QA mode only)
                # Teacher sees full document + question + answer (uncompressed)
                # and we extract only the Q+A tail for distillation targets.
                if teacher_model is not None and mode == "qa":
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
                        # Slice off the document prefix — keep only Q+A region
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

            running_loss += losses["total"].detach().item()
            micro_steps += 1

            # accumulate loss components
            for k, v in losses.items():
                if k == "total":
                    continue
                val = v.detach().item() if torch.is_tensor(v) else v
                running_components[k] = running_components.get(k, 0.0) + val

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
                            f"[{mode.upper()}] step {completed_steps}/{tcfg.max_steps}  "
                            f"loss={avg_loss:.4f}  ppl={ppl:.2f}  lr={lr:.2e}")

                        # log to wandb via accelerator
                        log_dict = {
                            f"{mode}/loss": avg_loss,
                            f"{mode}/ppl": ppl,
                            f"{mode}/lr": lr,
                        }
                        for k, v in running_components.items():
                            log_dict[f"{mode}/{k}"] = v / max(micro_steps, 1)
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
                    if mode == "ntp":
                        metrics = evaluate_ntp(model, eval_loader,
                                               accelerator)
                        if accelerator.is_main_process:
                            logger.info(
                                f"[NTP EVAL] step {completed_steps}  "
                                f"perplexity={metrics['perplexity']:.2f}  "
                                f"loss={metrics['loss']:.4f}")
                    else:
                        metrics = evaluate_qa(
                            model, eval_loader, tokenizer, accelerator,
                            max_new_tokens=config.qwen.max_answer_tokens)
                        if accelerator.is_main_process:
                            logger.info(
                                f"[QA EVAL] step {completed_steps}  "
                                f"EM={metrics['exact_match']:.2%}  "
                                f"F1={metrics['f1']:.4f}")

                    # log eval metrics
                    if accelerator.is_main_process:
                        accelerator.log(
                            {f"eval/{k}": v for k, v in metrics.items()},
                            step=completed_steps,
                        )

                    last_metrics = metrics
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

    # Logging: only main process gets INFO
    if accelerator.is_main_process:
        logging.basicConfig(level=logging.INFO,
                            format="%(asctime)s - %(name)s - %(message)s")
    else:
        logging.basicConfig(level=logging.WARNING)

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
            if getattr(wandb_conf, "offline", False):
                os.environ["WANDB_MODE"] = "offline"
            init_kwargs["wandb"] = wb_init

        accelerator.init_trackers(
            project_name=getattr(wandb_conf, "project", "deep-compressor"),
            config=config.to_dict(),
            init_kwargs=init_kwargs,
        )

    # ── tokenizer ──
    tokenizer = AutoTokenizer.from_pretrained(config.qwen.model_name_or_path)
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

    # ── dataloader ──
    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
    # MPS doesn't support multi-process data loading reliably
    num_workers = 0 if accelerator.device.type == "mps" else 2
    pin_memory = accelerator.device.type == "cuda"

    os.makedirs(tcfg.output_dir, exist_ok=True)

    last_metrics = None

    if tcfg.stage == 1:
        dataset = NTPDataset(data_path, tokenizer,
                             max_doc_tokens=config.qwen.max_doc_tokens,
                             segment_len=tcfg.ntp_segment_len)

        # Truncate training data if requested
        if max_train_samples > 0 and max_train_samples < len(dataset):
            dataset = Subset(dataset,
                             list(range(max_train_samples)))
            logger.info(f"Truncated NTP dataset to {max_train_samples} samples")

        # Split last 5000 samples (or 10%) as validation
        n_total = len(dataset)
        n_val = min(5000, n_total // 10) if n_total > 10 else 0
        n_train = n_total - n_val

        if n_val > 0:
            train_indices = list(range(n_train))
            val_indices = list(range(n_train, n_total))
            train_subset = Subset(dataset, train_indices)
            val_subset = Subset(dataset, val_indices)
        else:
            train_subset = dataset
            val_subset = None

        loader = DataLoader(train_subset, batch_size=tcfg.batch_size,
                            shuffle=True, collate_fn=collator,
                            num_workers=num_workers, pin_memory=pin_memory)
        eval_loader = None
        if val_subset is not None:
            if max_eval_samples > 0:
                val_subset = Subset(val_subset,
                                    list(range(min(max_eval_samples,
                                                   len(val_subset)))))
            eval_loader = DataLoader(val_subset, batch_size=tcfg.batch_size,
                                     shuffle=False, collate_fn=collator,
                                     num_workers=num_workers,
                                     pin_memory=pin_memory)
            logger.info(f"NTP eval split: {len(val_subset):,} samples")

        logger.info(f"NTP dataset: {len(train_subset):,} train samples, "
                    f"segment_len={tcfg.ntp_segment_len}")
        _, last_metrics = train_stage(
            config, model, loader, accelerator, mode="ntp",
            eval_loader=eval_loader, tokenizer=tokenizer,
            optuna_callback=optuna_callback,
            diagnostic_callback=diagnostic_callback)

    elif tcfg.stage == 2:
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

        _, last_metrics = train_stage(
            config, model, loader, accelerator, mode="qa",
            eval_loader=eval_loader, tokenizer=tokenizer,
            optuna_callback=optuna_callback,
            diagnostic_callback=diagnostic_callback)

    else:
        raise ValueError(f"Unknown stage: {tcfg.stage}")

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
    parser.add_argument("--stage", type=int, default=None,
                        help="Override training stage (1 or 2)")
    parser.add_argument("--resume_from", type=str, default=None,
                        help="Path to checkpoint dir")
    parser.add_argument("--eval_data_path", type=str, default=None,
                        help="Path to QA dev set (Stage 2) or NTP val split")
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
                        help="wandb run name")
    parser.add_argument("--wandb_offline", action="store_true",
                        help="Use wandb in offline mode")
    # Diagnostic callback options
    parser.add_argument("--diagnostic_every", type=int, default=0,
                        help="Run mid-training diagnostics every N steps (0 = off)")
    parser.add_argument("--diagnostic_data_path", type=str, default=None,
                        help="Data path for diagnostic callback (defaults to --data_path)")
    parser.add_argument("--diagnostic_experiments", type=str, default="4,5",
                        help="Comma-separated mid-training experiments to run")
    args = parser.parse_args()

    config = DeepCompressorConfig.from_yaml(args.config)
    if args.stage is not None:
        config.training.stage = args.stage

    # Build wandb config from CLI flags
    wandb_conf = None
    if args.wandb:
        from types import SimpleNamespace
        wandb_conf = SimpleNamespace(
            enabled=True,
            project=args.wandb_project,
            entity=args.wandb_entity,
            run_name=args.wandb_run_name,
            tags=[],
            offline=args.wandb_offline,
        )

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
