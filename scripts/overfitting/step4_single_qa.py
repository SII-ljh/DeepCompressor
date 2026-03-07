#!/usr/bin/env python3
"""Step 4: Single-sample two-phase QA overfit test — MacBook local validation (~10 min).

Phase 1: NTP pretrain on QA context text until PPL < 40 (gate).
Phase 2: QA fine-tune from Phase 1 checkpoint on same sample.

Minimal model: identity projections, Stage A only (1 cross + 1 self),
Stage B/C disabled. ~29.5M trainable params.

Goal: Phase 1 PPL < 40, Phase 2 QA CE loss < 0.5.

Usage:
  python scripts/overfitting/step4_single_qa.py
  python scripts/overfitting/step4_single_qa.py --data_path data/qa_tiny_train.json
  python scripts/overfitting/step4_single_qa.py --ntp_steps 500 --qa_steps 500 --lr 1e-3
"""

import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import torch
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deep_compressor.config import (
    AblationConfig,
    DeepCompressorConfig,
    LossConfig,
    PerceiverConfig,
    ProjectionConfig,
    QwenConfig,
    TrainingConfig,
)
from deep_compressor.data import PaddingCollator, QADataset
from deep_compressor.model import DeepCompressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("step4_single_qa")


# ── QA Context NTP Dataset ───────────────────────────────────────────

class QAContextNTPDataset(Dataset):
    """Reads QA JSON, uses 'context' field as NTP text."""

    def __init__(self, data_path, tokenizer, max_doc_tokens, segment_len, seed=42):
        with open(data_path) as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_doc_tokens = max_doc_tokens
        self.segment_len = segment_len
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]["context"]
        tokens = self.tokenizer(
            text, truncation=True,
            max_length=self.max_doc_tokens + self.segment_len,
            return_tensors="pt", padding=False,
        )
        input_ids = tokens["input_ids"].squeeze(0)
        total_len = input_ids.shape[0]

        if total_len <= self.segment_len + 1:
            split_point = total_len // 2
        else:
            max_split = total_len - self.segment_len
            split_point = self.rng.randint(1, max(1, min(max_split, self.max_doc_tokens)))

        return {
            "doc_input_ids": input_ids[:split_point],
            "segment_ids": input_ids[split_point:],
            "segment_labels": input_ids[split_point:].clone(),
        }


# ── Config ────────────────────────────────────────────────────────────

def build_ntp_config(args) -> DeepCompressorConfig:
    """Minimal model config for Phase 1 (NTP)."""
    return DeepCompressorConfig(
        qwen=QwenConfig(
            model_name_or_path="models/Qwen3-0.6B",
            max_doc_tokens=args.max_doc_tokens,
        ),
        perceiver=PerceiverConfig(
            perceiver_dim=1024,
            num_queries=args.num_queries,
            num_heads=16,
            head_dim=64,
            stage_a_cross_layers=2,
            stage_a_self_layers=2,
            stage_b_layers=2,
            stage_c_cross_layers=2,
            stage_c_self_layers=4,
            ff_mult=4,
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
        ),
        training=TrainingConfig(
            stage=1,
            learning_rate=args.lr,
            batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=args.ntp_steps,
            warmup_steps=args.warmup,
            weight_decay=0.0,
            max_grad_norm=1.0,
            scheduler=args.scheduler,
            seed=42,
            log_every=args.log_every,
            eval_every=args.eval_every,
            save_every=args.ntp_steps,  # save at end
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


def build_qa_config(args) -> DeepCompressorConfig:
    """Minimal model config for Phase 2 (QA). Enables question conditioning."""
    return DeepCompressorConfig(
        qwen=QwenConfig(
            model_name_or_path="models/Qwen3-0.6B",
            max_doc_tokens=args.max_doc_tokens,
        ),
        perceiver=PerceiverConfig(
            perceiver_dim=1024,
            num_queries=args.num_queries,
            num_heads=16,
            head_dim=64,
            stage_a_cross_layers=2,
            stage_a_self_layers=2,
            stage_b_layers=2,
            stage_c_cross_layers=2,
            stage_c_self_layers=4,
            ff_mult=4,
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
        ),
        training=TrainingConfig(
            stage=2,
            learning_rate=args.lr,
            batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=args.qa_steps,
            warmup_steps=args.warmup,
            weight_decay=0.0,
            max_grad_norm=1.0,
            scheduler=args.scheduler,
            seed=42,
            log_every=args.log_every,
            eval_every=args.eval_every,
            save_every=args.qa_steps,  # save at end
            output_dir=args.output_dir,
            ntp_segment_len=args.segment_len,
            gradient_checkpointing=True,
            mixed_precision="no",
        ),
        ablation=AblationConfig(
            down_proj_mode="identity",
            up_proj_mode="identity",
            query_condition_on_question=True,
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


@torch.no_grad()
def evaluate_qa(model, loader, device, use_bf16=False):
    """Compute average QA CE loss."""
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
        with autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            losses = model(
                mode="qa",
                doc_input_ids=batch_dev["doc_input_ids"],
                doc_attention_mask=batch_dev["doc_attention_mask"],
                q_input_ids=batch_dev["q_input_ids"],
                q_attention_mask=batch_dev["q_attention_mask"],
                answer_ids=batch_dev["answer_ids"],
                answer_attention_mask=batch_dev["answer_attention_mask"],
                answer_labels=batch_dev["answer_labels"],
            )
        bs = batch_dev["doc_input_ids"].shape[0]
        total_loss += losses["total"].float().item() * bs
        total_samples += bs

    avg_loss = total_loss / max(total_samples, 1)
    model.train()
    return {"loss": avg_loss}


# ── Training loops ────────────────────────────────────────────────────

def train_ntp_phase(model, train_loader, config, device, csv_writer, csv_file,
                    t0, use_bf16=False):
    """Phase 1: NTP pretraining. Returns final metrics."""
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

    step = 0
    epoch = 0
    running_loss = 0.0
    running_count = 0

    # Initial eval
    metrics = evaluate_ntp(model, train_loader, device, use_bf16=use_bf16)
    logger.info(f"[Phase1 init] loss={metrics['loss']:.4f}  ppl={metrics['perplexity']:.1f}")
    csv_writer.writerow(["ntp", 0, f"{metrics['loss']:.4f}", f"{metrics['perplexity']:.1f}",
                         "", f"{tcfg.learning_rate:.2e}", "0"])
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
                    f"[Phase1] step {step}/{tcfg.max_steps}  "
                    f"loss={avg:.4f}  ppl={ppl:.1f}  lr={lr:.2e}  "
                    f"elapsed={elapsed:.0f}s  epoch={epoch}"
                )
                csv_writer.writerow(["ntp", step, f"{avg:.4f}", f"{ppl:.1f}",
                                     "", f"{lr:.2e}", f"{elapsed:.0f}"])
                csv_file.flush()
                running_loss = 0.0
                running_count = 0

            if step >= tcfg.max_steps:
                break

    # Final eval
    metrics = evaluate_ntp(model, train_loader, device, use_bf16=use_bf16)
    elapsed = time.time() - t0
    logger.info(
        f"[Phase1 FINAL] loss={metrics['loss']:.4f}  ppl={metrics['perplexity']:.1f}"
    )
    csv_writer.writerow(["ntp", step, f"{metrics['loss']:.4f}",
                         f"{metrics['perplexity']:.1f}", "", "", f"{elapsed:.0f}"])
    csv_file.flush()

    return metrics


def train_qa_phase(model, train_loader, config, device, csv_writer, csv_file,
                   t0, use_bf16=False):
    """Phase 2: QA fine-tuning. Returns final metrics."""
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

    step = 0
    epoch = 0
    running_loss = 0.0
    running_count = 0

    # Initial eval
    metrics = evaluate_qa(model, train_loader, device, use_bf16=use_bf16)
    logger.info(f"[Phase2 init] qa_loss={metrics['loss']:.4f}")
    csv_writer.writerow(["qa", 0, "", "", f"{metrics['loss']:.4f}",
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
                    mode="qa",
                    doc_input_ids=batch_dev["doc_input_ids"],
                    doc_attention_mask=batch_dev["doc_attention_mask"],
                    q_input_ids=batch_dev["q_input_ids"],
                    q_attention_mask=batch_dev["q_attention_mask"],
                    answer_ids=batch_dev["answer_ids"],
                    answer_attention_mask=batch_dev["answer_attention_mask"],
                    answer_labels=batch_dev["answer_labels"],
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
                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - t0
                logger.info(
                    f"[Phase2] step {step}/{tcfg.max_steps}  "
                    f"qa_loss={avg:.4f}  lr={lr:.2e}  "
                    f"elapsed={elapsed:.0f}s  epoch={epoch}"
                )
                csv_writer.writerow(["qa", step, "", "", f"{avg:.4f}",
                                     f"{lr:.2e}", f"{elapsed:.0f}"])
                csv_file.flush()
                running_loss = 0.0
                running_count = 0

            if step >= tcfg.max_steps:
                break

    # Final eval
    metrics = evaluate_qa(model, train_loader, device, use_bf16=use_bf16)
    elapsed = time.time() - t0
    logger.info(f"[Phase2 FINAL] qa_loss={metrics['loss']:.4f}")
    csv_writer.writerow(["qa", step, "", "", f"{metrics['loss']:.4f}",
                         "", f"{elapsed:.0f}"])
    csv_file.flush()

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
        description="Step 4: Single-sample two-phase QA overfit (MacBook, ~10 min)")

    parser.add_argument("--data_path", type=str, default=None,
                        help="QA data path (default: auto-detect)")
    parser.add_argument("--max_doc_tokens", type=int, default=256,
                        help="Max document tokens (default: 256)")
    parser.add_argument("--segment_len", type=int, default=64,
                        help="NTP continuation segment length")
    parser.add_argument("--num_queries", type=int, default=32,
                        help="Number of Perceiver queries (default: 32)")

    parser.add_argument("--ntp_steps", type=int, default=300,
                        help="Phase 1 NTP training steps (default: 300)")
    parser.add_argument("--qa_steps", type=int, default=300,
                        help="Phase 2 QA training steps (default: 300)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Peak learning rate (default: 1e-3)")
    parser.add_argument("--warmup", type=int, default=20,
                        help="Warmup steps (default: 20)")
    parser.add_argument("--scheduler", type=str, default="cosine",
                        choices=["cosine", "constant"])
    parser.add_argument("--ppl_gate", type=float, default=40.0,
                        help="Phase 1 PPL gate — must reach this before Phase 2 (default: 40)")

    parser.add_argument("--log_every", type=int, default=10,
                        help="Log every N steps (default: 10)")
    parser.add_argument("--eval_every", type=int, default=100,
                        help="Eval every N steps (default: 100)")

    parser.add_argument("--output_dir", type=str,
                        default="outputs/overfit_step4")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Resume Phase 1 from checkpoint")

    args = parser.parse_args()

    # Auto-detect data path
    if args.data_path is None:
        candidates = [
            "data/qa_tiny_train.json",
            "data/qa_train.json",
        ]
        for p in candidates:
            if os.path.exists(p):
                args.data_path = p
                break
        if args.data_path is None:
            logger.error("No QA data found. Run: python scripts/prepare_data.py --make-tiny")
            sys.exit(1)

    # Device
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
    logger.info(f"Phase 1 (NTP): {args.ntp_steps} steps  |  Phase 2 (QA): {args.qa_steps} steps")
    logger.info(f"LR: {args.lr}  PPL gate: {args.ppl_gate}")
    logger.info(f"Doc tokens: {args.max_doc_tokens}  Segment len: {args.segment_len}")
    logger.info(f"Queries: {args.num_queries}  batch=1")

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)

    # CSV log (shared across both phases)
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "learning_curve.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["phase", "step", "ntp_loss", "ntp_ppl", "qa_loss", "lr", "elapsed_s"])

    t0 = time.time()

    # ── Phase 1: NTP on QA context ────────────────────────────────────

    logger.info("=" * 60)
    logger.info("  PHASE 1: NTP pretrain on QA context")
    logger.info("=" * 60)

    ntp_ds = QAContextNTPDataset(
        args.data_path, tokenizer,
        max_doc_tokens=args.max_doc_tokens,
        segment_len=args.segment_len,
    )
    ntp_train_ds = Subset(ntp_ds, [0])  # single sample
    ntp_loader = DataLoader(
        ntp_train_ds, batch_size=1, shuffle=False,
        collate_fn=collator, num_workers=0, pin_memory=False,
    )
    logger.info(f"NTP dataset: {len(ntp_ds)} total, using 1 sample for overfitting")

    ntp_config = build_ntp_config(args)
    logger.info("Loading Qwen3-0.6B + DeepCompressor (minimal model)...")
    model = DeepCompressor(ntp_config)

    if args.checkpoint:
        weights = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=False)
        logger.info(f"Loaded checkpoint: {args.checkpoint}")

    model = model.to(device)

    if ntp_config.training.gradient_checkpointing:
        model.qwen.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        logger.info("Gradient checkpointing enabled for Qwen")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {trainable:,} trainable / {total_params:,} total "
                f"({100*trainable/total_params:.1f}%)")

    ntp_metrics = train_ntp_phase(model, ntp_loader, ntp_config, device,
                                  csv_writer, csv_file, t0, use_bf16=use_bf16)

    # PPL gate
    phase1_ppl = ntp_metrics["perplexity"]
    phase1_ckpt = os.path.join(args.output_dir, "phase1_final.pt")
    _save_trainable(model, phase1_ckpt)

    if phase1_ppl >= args.ppl_gate:
        csv_file.close()
        elapsed = time.time() - t0
        print("\n" + "=" * 60)
        print("  STEP 4: SINGLE-SAMPLE TWO-PHASE QA OVERFIT")
        print("=" * 60)
        print(f"  Phase 1 PPL: {phase1_ppl:.1f} (gate: {args.ppl_gate})")
        print(f"\n  FAIL: Phase 1 PPL >= {args.ppl_gate} — NTP did not converge.")
        print(f"  Phase 2 skipped. Try more steps or higher LR.")
        print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")
        print("=" * 60)
        sys.exit(1)

    logger.info(f"Phase 1 gate PASSED: PPL {phase1_ppl:.1f} < {args.ppl_gate}")

    # ── Phase 2: QA fine-tune ─────────────────────────────────────────

    logger.info("=" * 60)
    logger.info("  PHASE 2: QA fine-tune from Phase 1 checkpoint")
    logger.info("=" * 60)

    # Rebuild model with QA config (enables question conditioning)
    qa_config = build_qa_config(args)
    model_qa = DeepCompressor(qa_config)

    # Load Phase 1 weights (strict=False: QA config may have new params)
    weights = torch.load(phase1_ckpt, map_location="cpu", weights_only=True)
    model_qa.load_state_dict(weights, strict=False)
    logger.info(f"Loaded Phase 1 checkpoint: {phase1_ckpt}")

    model_qa = model_qa.to(device)

    if qa_config.training.gradient_checkpointing:
        model_qa.qwen.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    # QA dataset: single sample
    qa_ds = QADataset(
        args.data_path, tokenizer,
        max_doc_tokens=args.max_doc_tokens,
    )
    qa_train_ds = Subset(qa_ds, [0])
    qa_loader = DataLoader(
        qa_train_ds, batch_size=1, shuffle=False,
        collate_fn=collator, num_workers=0, pin_memory=False,
    )
    logger.info(f"QA dataset: {len(qa_ds)} total, using 1 sample for overfitting")

    qa_metrics = train_qa_phase(model_qa, qa_loader, qa_config, device,
                                csv_writer, csv_file, t0, use_bf16=use_bf16)

    qa_ckpt = os.path.join(args.output_dir, "phase2_final.pt")
    _save_trainable(model_qa, qa_ckpt)

    csv_file.close()
    elapsed = time.time() - t0

    # ── Summary ───────────────────────────────────────────────────────

    phase2_loss = qa_metrics["loss"]

    print("\n" + "=" * 60)
    print("  STEP 4: SINGLE-SAMPLE TWO-PHASE QA OVERFIT")
    print("=" * 60)
    print(f"  Device:       {device} {'(bf16)' if use_bf16 else '(fp32)'}")
    print(f"  Model:        identity proj, Stage A only (1+1 layers)")
    print(f"  Trainable:    {trainable:,} params")
    print(f"  Data:         {args.data_path} (1 sample)")
    print(f"  Phase 1:      {args.ntp_steps} NTP steps -> PPL {phase1_ppl:.1f}")
    print(f"  Phase 2:      {args.qa_steps} QA steps -> loss {phase2_loss:.4f}")
    print(f"  Elapsed:      {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"  Output:       {args.output_dir}/")
    print(f"  Curve CSV:    {csv_path}")

    if phase2_loss < 0.5:
        print("\n  PASS: QA loss < 0.5 — model can overfit a single QA sample")
    elif phase2_loss < 2.0:
        print("\n  PARTIAL: QA loss 0.5-2.0 — learning but not fully memorized, try more steps")
    else:
        print("\n  FAIL: QA loss > 2.0 — QA path not learning, investigate")
    print("=" * 60)


if __name__ == "__main__":
    main()
