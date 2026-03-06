#!/usr/bin/env python3
"""Step 1: Single-sample overfit test — MacBook local validation (~5 min).

Minimal model: identity projections, Stage A only (1 cross + 1 self),
Stage B/C disabled. ~29.5M trainable params.

Goal: loss drops from ~11 to <1, PPL approaches 1.

Usage:
  python scripts/overfitting/step1_single_sample.py
  python scripts/overfitting/step1_single_sample.py --steps 500 --lr 1e-3
  python scripts/overfitting/step1_single_sample.py --data_path data/ntp_tiny.jsonl
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
logger = logging.getLogger("step1_single_sample")


# ── Config ────────────────────────────────────────────────────────────

def build_config(args) -> DeepCompressorConfig:
    """Minimal model for single-sample overfitting."""
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
            num_queries=args.num_queries,
            num_heads=16,
            head_dim=64,
            stage_a_cross_layers=2,  # overridden by ablation
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
            batch_size=1,
            gradient_accumulation_steps=1,
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
        ablation=AblationConfig(
            down_proj_mode="identity",
            up_proj_mode="identity",
            query_condition_on_question=False,
            enable_stage_a=True,
            enable_stage_b=False,
            enable_stage_c=False,
            override_stage_a_cross_layers=1,
            override_stage_a_self_layers=1,
        ),
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

def train(model, train_loader, config, device, args, use_bf16=False):
    """Single-sample NTP training loop — no val split."""
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
    csv_writer.writerow(["step", "train_loss", "train_ppl", "lr", "elapsed_s"])

    step = 0
    epoch = 0
    running_loss = 0.0
    running_count = 0
    best_loss = float("inf")
    t0 = time.time()

    if use_bf16:
        logger.info("Using bf16 mixed precision (autocast)")

    # Initial eval on training sample
    metrics = evaluate_ntp(model, train_loader, device, use_bf16=use_bf16)
    logger.info(f"[init] train_loss={metrics['loss']:.4f}  train_ppl={metrics['perplexity']:.1f}")
    csv_writer.writerow([0, f"{metrics['loss']:.4f}", f"{metrics['perplexity']:.1f}",
                         f"{tcfg.learning_rate:.2e}", "0"])
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

            loss = losses["total"]
            loss.backward()
            running_loss += losses["total"].float().item()
            running_count += 1

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                tcfg.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1

            if step % tcfg.log_every == 0:
                avg = running_loss / running_count
                ppl = math.exp(min(avg, 20))
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                logger.info(
                    f"step {step}/{tcfg.max_steps}  "
                    f"loss={avg:.4f}  ppl={ppl:.1f}  lr={lr:.2e}  "
                    f"elapsed={elapsed:.0f}s  epoch={epoch}"
                )
                csv_writer.writerow([step, f"{avg:.4f}", f"{ppl:.1f}",
                                     f"{lr:.2e}", f"{elapsed:.0f}"])
                csv_file.flush()

                if avg < best_loss:
                    best_loss = avg

                running_loss = 0.0
                running_count = 0

            if step % tcfg.save_every == 0:
                _save_trainable(model, os.path.join(args.output_dir, f"step_{step}.pt"))

            if step >= tcfg.max_steps:
                break

    # Final eval
    metrics = evaluate_ntp(model, train_loader, device, use_bf16=use_bf16)
    elapsed = time.time() - t0
    logger.info(
        f"[FINAL] train_loss={metrics['loss']:.4f}  "
        f"train_ppl={metrics['perplexity']:.1f}  best_loss={best_loss:.4f}"
    )
    csv_writer.writerow([step, f"{metrics['loss']:.4f}", f"{metrics['perplexity']:.1f}",
                         "", f"{elapsed:.0f}"])
    csv_file.close()
    _save_trainable(model, os.path.join(args.output_dir, "final.pt"))

    logger.info(f"Training done. {step} steps in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    logger.info(f"Learning curve saved to: {csv_path}")

    return metrics


def _save_trainable(model, path):
    """Save only trainable weights."""
    state = {k: v.cpu() for k, v in model.state_dict().items()
             if not k.startswith("qwen.")}
    torch.save(state, path)
    logger.info(f"  Saved checkpoint: {path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 1: Single-sample overfit (MacBook, ~5 min)")

    parser.add_argument("--data_path", type=str, default=None,
                        help="NTP data path (default: auto-detect)")
    parser.add_argument("--max_doc_tokens", type=int, default=256,
                        help="Max document tokens (default: 256)")
    parser.add_argument("--segment_len", type=int, default=64,
                        help="NTP continuation segment length")
    parser.add_argument("--num_queries", type=int, default=32,
                        help="Number of Perceiver queries (default: 32)")

    parser.add_argument("--steps", type=int, default=300,
                        help="Total training steps (default: 300)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Peak learning rate (default: 1e-3)")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Warmup steps (default: 20)")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "constant"])

    parser.add_argument("--log_every", type=int, default=10,
                        help="Log every N steps (default: 10)")
    parser.add_argument("--eval_every", type=int, default=100,
                        help="Eval every N steps (default: 100)")
    parser.add_argument("--save_every", type=int, default=100,
                        help="Save every N steps (default: 100)")

    parser.add_argument("--output_dir", type=str,
                        default="outputs/overfit_step1")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume from checkpoint")

    args = parser.parse_args()

    # Auto-detect data path
    if args.data_path is None:
        candidates = [
            "data/ntp_tiny.jsonl",
            "data/ablation/ntp_ablation.jsonl",
            "data/ntp_train.jsonl",
        ]
        for p in candidates:
            if os.path.exists(p):
                args.data_path = p
                break
        if args.data_path is None:
            logger.error("No NTP data found. Run: python scripts/prepare_data.py --make-tiny")
            sys.exit(1)

    # Device: prefer MPS/CPU for MacBook
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    use_bf16 = (device.type == "cuda" and torch.cuda.is_bf16_supported())

    logger.info(f"Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.0f} GB)")
    logger.info(f"bf16: {'ON' if use_bf16 else 'OFF'}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Steps: {args.steps}  LR: {args.lr}")
    logger.info(f"Doc tokens: {args.max_doc_tokens}  Segment len: {args.segment_len}")
    logger.info(f"Queries: {args.num_queries}  batch=1  grad_accum=1")

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Dataset: use only the first sample
    full_ds = NTPDataset(
        args.data_path, tokenizer,
        max_doc_tokens=args.max_doc_tokens,
        segment_len=args.segment_len,
    )
    train_ds = Subset(full_ds, [0])  # single sample
    logger.info(f"Dataset: {len(full_ds)} total, using 1 sample for overfitting")

    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_ds, batch_size=1, shuffle=False,
        collate_fn=collator, num_workers=0, pin_memory=False,
    )

    # Model
    config = build_config(args)
    logger.info("Loading Qwen3-0.6B + DeepCompressor (minimal model)...")
    model = DeepCompressor(config)

    if args.checkpoint:
        weights = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=False)
        logger.info(f"Loaded checkpoint: {args.checkpoint}")

    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {trainable:,} trainable / {total:,} total "
                f"({100*trainable/total:.1f}%)")

    # Train
    metrics = train(model, train_loader, config, device, args, use_bf16=use_bf16)

    # Summary
    print("\n" + "=" * 60)
    print("  STEP 1: SINGLE-SAMPLE OVERFIT TEST")
    print("=" * 60)
    print(f"  Device:       {device} {'(bf16)' if use_bf16 else '(fp32)'}")
    print(f"  Model:        identity proj, Stage A only (1+1 layers)")
    print(f"  Trainable:    {trainable:,} params")
    print(f"  Steps:        {args.steps}")
    print(f"  Final loss:   {metrics['loss']:.4f}")
    print(f"  Final PPL:    {metrics['perplexity']:.1f}")
    print(f"  Output:       {args.output_dir}/")

    if metrics["perplexity"] < 5:
        print("\n  PASS: PPL < 5 — model can memorize a single sample")
    elif metrics["perplexity"] < 50:
        print("\n  PARTIAL: PPL 5-50 — learning but not fully memorized, try more steps")
    else:
        print("\n  FAIL: PPL > 50 — model cannot learn, investigate gradient flow")
    print("=" * 60)


if __name__ == "__main__":
    main()
