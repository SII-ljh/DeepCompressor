#!/usr/bin/env python3
"""Step 3: Full-model overfit on 5% ablation data — H200 validation (~1-2 hr).

Full architecture: MLP projections, all Perceiver stages enabled,
default layer counts. ~172M trainable params.

Goal: val PPL < 50.

Usage:
  python scripts/overfitting/step3_ablation_full.py
  python scripts/overfitting/step3_ablation_full.py --steps 30000 --lr 5e-4
  python scripts/overfitting/step3_ablation_full.py --batch_size 64 --max_doc_tokens 2048
"""

import argparse
import csv
import logging
import math
import os
import sys
import time
from pathlib import Path

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deep_compressor.config import (
    AblationConfig,
    DeepCompressorConfig,
    FinBERTConfig,
    LossConfig,
    PerceiverConfig,
    ProjectionConfig,
    QwenConfig,
    TrainingConfig,
)
from deep_compressor.data import NTPDataset, PaddingCollator
from deep_compressor.model import DeepCompressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("step3_ablation_full")


# ── Config ────────────────────────────────────────────────────────────

def build_config(args) -> DeepCompressorConfig:
    """Full model config — same as overfit_test.py."""
    return DeepCompressorConfig(
        qwen=QwenConfig(
            model_name_or_path="models/Qwen3-0.6B",
            hidden_size=1024,
            num_hidden_layers=28,
            vocab_size=151936,
            max_doc_tokens=args.max_doc_tokens,
        ),
        finbert=FinBERTConfig(enabled=False),
        perceiver=PerceiverConfig(
            perceiver_dim=1024,
            num_queries=64,
            num_heads=16,
            head_dim=64,
            stage_a_cross_layers=2,
            stage_a_self_layers=2,
            stage_b_layers=2,
            stage_c_cross_layers=2,
            stage_c_self_layers=4,
            ff_mult=4,
            anchor_score_scale_init=1.0,
            dropout=0.0,
        ),
        projection=ProjectionConfig(
            down_hidden=768,
            up_hidden=768,
            dropout=0.0,
        ),
        loss=LossConfig(
            qa_ce_weight=1.0,
            kl_weight=0.0,
            hidden_mse_weight=0.0,
            anchor_recon_weight=0.0,
        ),
        training=TrainingConfig(
            stage=1,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_steps=args.steps,
            warmup_steps=args.warmup,
            weight_decay=0.0,
            max_grad_norm=1.0,
            scheduler=args.scheduler,
            seed=42,
            log_every=args.log_every,
            eval_every=args.eval_every,
            save_every=args.save_every,
            output_dir=args.output_dir,
            ntp_segment_len=args.segment_len,
            gradient_checkpointing=True,
            mixed_precision="no",
        ),
        # Full architecture — no ablation overrides
        ablation=AblationConfig(),
    )


# ── Evaluation ────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_ntp(model, loader, device, use_bf16=False):
    """Compute average NTP loss and perplexity."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
        with autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            losses = model(
                mode="ntp",
                doc_input_ids=batch_dev["doc_input_ids"],
                doc_attention_mask=batch_dev["doc_attention_mask"],
                segment_ids=batch_dev["segment_ids"],
                segment_attention_mask=batch_dev["segment_attention_mask"],
                segment_labels=batch_dev["segment_labels"],
            )
        bs = batch_dev["doc_input_ids"].shape[0]
        total_loss += losses["total"].float().item() * bs
        total_samples += bs

    avg_loss = total_loss / max(total_samples, 1)
    ppl = math.exp(min(avg_loss, 20))
    model.train()
    return {"loss": avg_loss, "perplexity": ppl}


# ── Training loop ─────────────────────────────────────────────────────

def train(model, train_loader, val_loader, config, device, args, use_bf16=False):
    """NTP training loop with periodic val PPL evaluation."""
    tcfg = config.training

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=tcfg.learning_rate, weight_decay=tcfg.weight_decay,
    )

    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(step):
        if step < tcfg.warmup_steps:
            return step / max(1, tcfg.warmup_steps)
        if tcfg.scheduler == "cosine":
            progress = (step - tcfg.warmup_steps) / max(1, tcfg.max_steps - tcfg.warmup_steps)
            return 0.5 * (1 + math.cos(math.pi * progress))
        return 1.0

    scheduler = LambdaLR(optimizer, lr_lambda)

    csv_path = os.path.join(args.output_dir, "learning_curve.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "train_loss", "val_loss", "val_ppl", "lr", "elapsed_s"])

    step = 0
    epoch = 0
    running_loss = 0.0
    micro_steps = 0
    best_ppl = float("inf")
    t0 = time.time()

    if use_bf16:
        logger.info("Using bf16 mixed precision (autocast)")

    # Initial eval
    if val_loader is not None:
        metrics = evaluate_ntp(model, val_loader, device, use_bf16=use_bf16)
        logger.info(f"[init] val_loss={metrics['loss']:.4f}  val_ppl={metrics['perplexity']:.1f}")
        csv_writer.writerow([0, "", f"{metrics['loss']:.4f}",
                             f"{metrics['perplexity']:.1f}", f"{tcfg.learning_rate:.2e}", "0"])
        csv_file.flush()

    model.train()

    while step < tcfg.max_steps:
        epoch += 1
        for batch in train_loader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

            with autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                losses = model(
                    mode="ntp",
                    doc_input_ids=batch_dev["doc_input_ids"],
                    doc_attention_mask=batch_dev["doc_attention_mask"],
                    segment_ids=batch_dev["segment_ids"],
                    segment_attention_mask=batch_dev["segment_attention_mask"],
                    segment_labels=batch_dev["segment_labels"],
                )

            loss = losses["total"] / tcfg.gradient_accumulation_steps
            loss.backward()
            running_loss += losses["total"].float().item()
            micro_steps += 1

            if micro_steps % tcfg.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    tcfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

                if step % tcfg.log_every == 0:
                    avg = running_loss / micro_steps
                    ppl = math.exp(min(avg, 20))
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    logger.info(
                        f"step {step}/{tcfg.max_steps}  "
                        f"loss={avg:.4f}  ppl={ppl:.1f}  lr={lr:.2e}  "
                        f"elapsed={elapsed:.0f}s  epoch={epoch}"
                    )
                    running_loss = 0.0
                    micro_steps = 0

                if val_loader is not None and step % tcfg.eval_every == 0:
                    metrics = evaluate_ntp(model, val_loader, device, use_bf16=use_bf16)
                    elapsed = time.time() - t0
                    lr = scheduler.get_last_lr()[0]

                    improved = ""
                    if metrics["perplexity"] < best_ppl:
                        best_ppl = metrics["perplexity"]
                        improved = " ** best **"
                        _save_trainable(model, os.path.join(args.output_dir, "best.pt"))

                    logger.info(
                        f"[EVAL] step {step}  val_loss={metrics['loss']:.4f}  "
                        f"val_ppl={metrics['perplexity']:.1f}  "
                        f"best_ppl={best_ppl:.1f}{improved}"
                    )
                    csv_writer.writerow([
                        step, "", f"{metrics['loss']:.4f}",
                        f"{metrics['perplexity']:.1f}", f"{lr:.2e}",
                        f"{elapsed:.0f}",
                    ])
                    csv_file.flush()
                    model.train()

                if step % tcfg.save_every == 0:
                    _save_trainable(model, os.path.join(args.output_dir, f"step_{step}.pt"))

                if step >= tcfg.max_steps:
                    break

    # Final eval
    if val_loader is not None:
        metrics = evaluate_ntp(model, val_loader, device, use_bf16=use_bf16)
        elapsed = time.time() - t0
        if metrics["perplexity"] < best_ppl:
            best_ppl = metrics["perplexity"]
            _save_trainable(model, os.path.join(args.output_dir, "best.pt"))
        logger.info(
            f"[FINAL] val_loss={metrics['loss']:.4f}  "
            f"val_ppl={metrics['perplexity']:.1f}  best_ppl={best_ppl:.1f}"
        )
        csv_writer.writerow([
            step, "", f"{metrics['loss']:.4f}",
            f"{metrics['perplexity']:.1f}", "", f"{elapsed:.0f}",
        ])

    csv_file.close()
    _save_trainable(model, os.path.join(args.output_dir, "final.pt"))

    elapsed = time.time() - t0
    logger.info(f"Training done. {step} steps in {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    logger.info(f"Best val PPL: {best_ppl:.1f}")
    logger.info(f"Learning curve saved to: {csv_path}")

    return best_ppl


def _save_trainable(model, path):
    """Save only trainable weights."""
    state = {k: v.cpu() for k, v in model.state_dict().items()
             if not k.startswith("qwen.")}
    torch.save(state, path)
    logger.info(f"  Saved checkpoint: {path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 3: Full-model overfit on ablation data (H200, ~1-2 hr)")

    parser.add_argument("--data_path", type=str, default=None,
                        help="NTP data path (default: auto-detect ablation subset)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of data for validation (default: 0.1)")
    parser.add_argument("--max_doc_tokens", type=int, default=1024,
                        help="Max document tokens (default: 1024)")
    parser.add_argument("--segment_len", type=int, default=128,
                        help="NTP continuation segment length (default: 128)")

    parser.add_argument("--steps", type=int, default=20000,
                        help="Total training steps (default: 20000)")
    parser.add_argument("--lr", type=float, default=5e-4,
                        help="Peak learning rate (default: 5e-4)")
    parser.add_argument("--batch_size", type=int, default=48,
                        help="Batch size (default: 48)")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps (default: 1)")
    parser.add_argument("--warmup", type=int, default=500,
                        help="Warmup steps (default: 500)")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "constant"])

    parser.add_argument("--log_every", type=int, default=50,
                        help="Log every N steps (default: 50)")
    parser.add_argument("--eval_every", type=int, default=500,
                        help="Evaluate val PPL every N steps (default: 500)")
    parser.add_argument("--save_every", type=int, default=5000,
                        help="Save every N steps (default: 5000)")

    parser.add_argument("--no_bf16", action="store_true",
                        help="Disable bf16 mixed precision")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader workers (default: 4 on CUDA)")

    parser.add_argument("--output_dir", type=str,
                        default="outputs/overfit_step3")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint")

    args = parser.parse_args()

    # Auto-detect data path
    if args.data_path is None:
        candidates = [
            "data/ablation/ntp_ablation.jsonl",
            "data/ntp_tiny.jsonl",
            "data/ntp_train.jsonl",
        ]
        for p in candidates:
            if os.path.exists(p):
                args.data_path = p
                break
        if args.data_path is None:
            logger.error("No NTP data found. Run: python scripts/prepare_data.py --make-ablation")
            sys.exit(1)

    # Device
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    use_bf16 = (device.type == "cuda" and not args.no_bf16
                and torch.cuda.is_bf16_supported())

    logger.info(f"Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.0f} GB)")
    logger.info(f"bf16: {'ON' if use_bf16 else 'OFF'}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Steps: {args.steps}  LR: {args.lr}  "
                f"Batch: {args.batch_size}x{args.grad_accum} "
                f"(effective {args.batch_size * args.grad_accum})")
    logger.info(f"Doc tokens: {args.max_doc_tokens}  Segment len: {args.segment_len}")

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset & split
    full_ds = NTPDataset(
        args.data_path, tokenizer,
        max_doc_tokens=args.max_doc_tokens,
        segment_len=args.segment_len,
    )
    n_total = len(full_ds)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val

    train_ds = Subset(full_ds, list(range(n_train)))
    val_ds = Subset(full_ds, list(range(n_train, n_total)))
    logger.info(f"Dataset: {n_total} total -> {n_train} train / {n_val} val")

    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
    if args.num_workers is not None:
        num_workers = args.num_workers
    else:
        num_workers = 4 if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=num_workers, pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=num_workers, pin_memory=pin_memory,
    )

    # Model
    config = build_config(args)
    logger.info("Loading Qwen3-0.6B + DeepCompressor (full model)...")
    model = DeepCompressor(config)

    if args.checkpoint:
        weights = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=False)
        logger.info(f"Loaded checkpoint: {args.checkpoint}")

    model = model.to(device)

    # Enable gradient checkpointing to reduce decoder activation memory
    if config.training.gradient_checkpointing:
        model.qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        logger.info("Gradient checkpointing enabled for Qwen")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {trainable:,} trainable / {total:,} total "
                f"({100*trainable/total:.1f}%)")

    # Train
    best_ppl = train(model, train_loader, val_loader, config, device, args,
                     use_bf16=use_bf16)

    # Summary
    print("\n" + "=" * 60)
    print("  STEP 3: FULL-MODEL OVERFIT ON ABLATION DATA")
    print("=" * 60)
    print(f"  Device:       {device} {'(bf16)' if use_bf16 else '(fp32)'}")
    print(f"  Model:        MLP projections, all stages, full layers")
    print(f"  Trainable:    {trainable:,} params")
    print(f"  Data:         {args.data_path} ({n_train} train / {n_val} val)")
    print(f"  Batch:        {args.batch_size}x{args.grad_accum} = "
          f"{args.batch_size * args.grad_accum} effective")
    print(f"  Steps:        {args.steps}")
    print(f"  Best PPL:     {best_ppl:.1f}")
    print(f"  Output:       {args.output_dir}/")
    print(f"  Curve CSV:    {args.output_dir}/learning_curve.csv")

    if best_ppl < 50:
        print("\n  PASS: val PPL < 50 — full compression pipeline works")
    elif best_ppl < 200:
        print("\n  PARTIAL: val PPL 50-200 — learning but needs more steps or tuning")
    else:
        print("\n  FAIL: val PPL > 200 — full model not converging, investigate")
    print("=" * 60)


if __name__ == "__main__":
    main()
