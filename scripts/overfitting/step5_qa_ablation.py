#!/usr/bin/env python3
"""Step 5: Two-phase QA overfit on 5% ablation data — H200 validation (~2-3 hr).

Phase 1: NTP pretrain on QA context texts until val PPL < 40 (gate).
Phase 2: QA fine-tune from Phase 1 checkpoint on same data.

Full architecture: identity projections, all Perceiver stages enabled,
default layer counts. Optional teacher distillation via --use_teacher.

Goal: Phase 1 val PPL < 40, Phase 2 val QA CE loss decreasing consistently.

Usage:
  python scripts/overfitting/step5_qa_ablation.py
  python scripts/overfitting/step5_qa_ablation.py --data_path data/qa_tiny_train.json
  python scripts/overfitting/step5_qa_ablation.py --ntp_steps 15000 --qa_steps 15000 --use_teacher
  python scripts/overfitting/step5_qa_ablation.py --batch_size 32 --lr 1e-3
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
from deep_compressor.eval import compute_exact_match, compute_f1
from deep_compressor.model import DeepCompressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("step5_qa_ablation")


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
    """Full model config for Phase 1 (NTP). All stages, identity projections."""
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
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_steps=args.ntp_steps,
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
        # Full architecture — all stages enabled, identity projections
        ablation=AblationConfig(
            down_proj_mode="identity",
            up_proj_mode="identity",
            query_condition_on_question=False,
        ),
    )


def build_qa_config(args) -> DeepCompressorConfig:
    """Full model config for Phase 2 (QA). Enables question conditioning."""
    kl_weight = 1.0 if args.use_teacher else 0.0
    hidden_mse_weight = 0.5 if args.use_teacher else 0.0

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
            kl_weight=kl_weight,
            hidden_mse_weight=hidden_mse_weight,
        ),
        training=TrainingConfig(
            stage=2,
            learning_rate=args.lr,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            max_steps=args.qa_steps,
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
        # Full architecture + question conditioning
        ablation=AblationConfig(
            down_proj_mode="identity",
            up_proj_mode="identity",
            query_condition_on_question=True,
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
    """Compute average QA CE loss (no teacher distillation during eval)."""
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


@torch.no_grad()
def evaluate_qa_generation(model, loader, tokenizer, device, use_bf16=False,
                           max_new_tokens=128, max_show=5):
    """Generate answers and compute EM/F1 against ground truth."""
    model.eval()
    all_em, all_f1 = [], []
    examples = []

    for batch in loader:
        batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

        with autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            byte_array = model.encode_document(
                batch_dev["doc_input_ids"], batch_dev["doc_attention_mask"])
            queries = model.encode_question(
                batch_dev["q_input_ids"], batch_dev["q_attention_mask"])
            latent = model.compress(
                queries, byte_array, byte_mask=batch_dev["doc_attention_mask"])
            prefix_embeds = model.up_mlp(latent)

            gen_ids = model.generate_answer(
                prefix_embeds, batch_dev["q_input_ids"],
                batch_dev["q_attention_mask"],
                tokenizer=tokenizer, max_new_tokens=max_new_tokens,
            )

        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        golds = batch_dev["answer_text"]

        for pred, gold in zip(preds, golds):
            em = compute_exact_match(pred, gold)
            f1 = compute_f1(pred, gold)
            all_em.append(em)
            all_f1.append(f1)
            if len(examples) < max_show:
                examples.append({"pred": pred, "gold": gold, "em": em, "f1": f1})

    n = max(len(all_em), 1)
    metrics = {
        "exact_match": sum(all_em) / n,
        "f1": sum(all_f1) / n,
        "n_samples": len(all_em),
    }
    model.train()
    return metrics, examples


# ── Teacher helper ────────────────────────────────────────────────────

@torch.no_grad()
def compute_teacher_outputs(teacher_model, batch_dev, device):
    """On-the-fly teacher forward: full doc + question + answer -> logits & hidden.

    Teacher sees uncompressed document, so its logits/hidden for the Q+A region
    serve as distillation targets.
    """
    doc_ids = batch_dev["doc_input_ids"]
    q_ids = batch_dev["q_input_ids"]
    answer_ids = batch_dev["answer_ids"]

    # Concatenate full sequence: doc + question + answer
    t_input_ids = torch.cat([doc_ids, q_ids, answer_ids], dim=1)
    t_attention_mask = torch.ones_like(t_input_ids)

    t_out = teacher_model(
        input_ids=t_input_ids,
        attention_mask=t_attention_mask,
        output_hidden_states=True,
    )

    # Extract Q+A region (skip document prefix)
    doc_len = doc_ids.shape[1]
    teacher_logits = t_out.logits[:, doc_len:, :].detach()
    teacher_hidden = [h[:, doc_len:, :].detach() for h in t_out.hidden_states]

    return teacher_logits, teacher_hidden


# ── Training loops ────────────────────────────────────────────────────

def train_ntp_phase(model, train_loader, val_loader, config, device,
                    csv_writer, csv_file, t0, use_bf16=False):
    """Phase 1: NTP pretraining with val PPL tracking. Returns best val PPL."""
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
    micro_steps = 0
    best_ppl = float("inf")

    # Initial eval
    if val_loader is not None:
        metrics = evaluate_ntp(model, val_loader, device, use_bf16=use_bf16)
        logger.info(f"[Phase1 init] val_loss={metrics['loss']:.4f}  val_ppl={metrics['perplexity']:.1f}")
        csv_writer.writerow(["ntp", 0, "", f"{metrics['loss']:.4f}",
                             f"{metrics['perplexity']:.1f}", "", f"{tcfg.learning_rate:.2e}", "0"])
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
                        f"[Phase1] step {step}/{tcfg.max_steps}  "
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
                        _save_trainable(model, os.path.join(
                            config.training.output_dir, "phase1_best.pt"))

                    logger.info(
                        f"[Phase1 EVAL] step {step}  val_loss={metrics['loss']:.4f}  "
                        f"val_ppl={metrics['perplexity']:.1f}  "
                        f"best_ppl={best_ppl:.1f}{improved}"
                    )
                    csv_writer.writerow([
                        "ntp", step, "", f"{metrics['loss']:.4f}",
                        f"{metrics['perplexity']:.1f}", "", f"{lr:.2e}",
                        f"{elapsed:.0f}",
                    ])
                    csv_file.flush()
                    model.train()

                if step % tcfg.save_every == 0:
                    _save_trainable(model, os.path.join(
                        config.training.output_dir, f"phase1_step_{step}.pt"))

                if step >= tcfg.max_steps:
                    break

    # Final eval
    if val_loader is not None:
        metrics = evaluate_ntp(model, val_loader, device, use_bf16=use_bf16)
        elapsed = time.time() - t0
        if metrics["perplexity"] < best_ppl:
            best_ppl = metrics["perplexity"]
            _save_trainable(model, os.path.join(
                config.training.output_dir, "phase1_best.pt"))
        logger.info(
            f"[Phase1 FINAL] val_loss={metrics['loss']:.4f}  "
            f"val_ppl={metrics['perplexity']:.1f}  best_ppl={best_ppl:.1f}"
        )
        csv_writer.writerow([
            "ntp", step, "", f"{metrics['loss']:.4f}",
            f"{metrics['perplexity']:.1f}", "", "", f"{elapsed:.0f}",
        ])
        csv_file.flush()

    _save_trainable(model, os.path.join(config.training.output_dir, "phase1_final.pt"))
    return best_ppl


def train_qa_phase(model, train_loader, val_loader, config, device,
                   csv_writer, csv_file, t0, use_bf16=False,
                   teacher_model=None):
    """Phase 2: QA fine-tuning with val loss tracking. Returns best val QA loss."""
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
    micro_steps = 0
    best_val_loss = float("inf")

    # Initial eval
    if val_loader is not None:
        metrics = evaluate_qa(model, val_loader, device, use_bf16=use_bf16)
        logger.info(f"[Phase2 init] val_qa_loss={metrics['loss']:.4f}")
        csv_writer.writerow(["qa", 0, "", "", "", f"{metrics['loss']:.4f}",
                             f"{tcfg.learning_rate:.2e}", "0"])
        csv_file.flush()

    model.train()

    while step < tcfg.max_steps:
        epoch += 1
        for batch in train_loader:
            batch_dev = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

            # Build forward kwargs
            fwd_kwargs = dict(
                mode="qa",
                doc_input_ids=batch_dev["doc_input_ids"],
                doc_attention_mask=batch_dev["doc_attention_mask"],
                q_input_ids=batch_dev["q_input_ids"],
                q_attention_mask=batch_dev["q_attention_mask"],
                answer_ids=batch_dev["answer_ids"],
                answer_attention_mask=batch_dev["answer_attention_mask"],
                answer_labels=batch_dev["answer_labels"],
                global_step=step,
            )

            # On-the-fly teacher distillation
            if teacher_model is not None:
                teacher_logits, teacher_hidden = compute_teacher_outputs(
                    teacher_model, batch_dev, device)
                fwd_kwargs["teacher_logits"] = teacher_logits
                fwd_kwargs["teacher_hidden"] = teacher_hidden

            with autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
                losses = model(**fwd_kwargs)

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
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - t0
                    loss_parts = []
                    if "qa_ce" in losses:
                        loss_parts.append(f"qa_ce={losses['qa_ce'].item():.4f}")
                    if "kl" in losses and losses["kl"] is not None:
                        loss_parts.append(f"kl={losses['kl'].item():.4f}")
                    if "hidden_mse" in losses and losses["hidden_mse"] is not None:
                        loss_parts.append(f"hmse={losses['hidden_mse'].item():.4f}")
                    detail = "  ".join(loss_parts) if loss_parts else ""
                    logger.info(
                        f"[Phase2] step {step}/{tcfg.max_steps}  "
                        f"total={avg:.4f}  {detail}  lr={lr:.2e}  "
                        f"elapsed={elapsed:.0f}s  epoch={epoch}"
                    )
                    running_loss = 0.0
                    micro_steps = 0

                if val_loader is not None and step % tcfg.eval_every == 0:
                    metrics = evaluate_qa(model, val_loader, device, use_bf16=use_bf16)
                    elapsed = time.time() - t0
                    lr = scheduler.get_last_lr()[0]

                    improved = ""
                    if metrics["loss"] < best_val_loss:
                        best_val_loss = metrics["loss"]
                        improved = " ** best **"
                        _save_trainable(model, os.path.join(
                            config.training.output_dir, "phase2_best.pt"))

                    logger.info(
                        f"[Phase2 EVAL] step {step}  val_qa_loss={metrics['loss']:.4f}  "
                        f"best={best_val_loss:.4f}{improved}"
                    )
                    csv_writer.writerow([
                        "qa", step, "", "", "", f"{metrics['loss']:.4f}",
                        f"{lr:.2e}", f"{elapsed:.0f}",
                    ])
                    csv_file.flush()
                    model.train()

                if step % tcfg.save_every == 0:
                    _save_trainable(model, os.path.join(
                        config.training.output_dir, f"phase2_step_{step}.pt"))

                if step >= tcfg.max_steps:
                    break

    # Final eval
    if val_loader is not None:
        metrics = evaluate_qa(model, val_loader, device, use_bf16=use_bf16)
        elapsed = time.time() - t0
        if metrics["loss"] < best_val_loss:
            best_val_loss = metrics["loss"]
            _save_trainable(model, os.path.join(
                config.training.output_dir, "phase2_best.pt"))
        logger.info(
            f"[Phase2 FINAL] val_qa_loss={metrics['loss']:.4f}  best={best_val_loss:.4f}"
        )
        csv_writer.writerow([
            "qa", step, "", "", "", f"{metrics['loss']:.4f}",
            "", f"{elapsed:.0f}",
        ])
        csv_file.flush()

    _save_trainable(model, os.path.join(config.training.output_dir, "phase2_final.pt"))
    return best_val_loss


def _save_trainable(model, path):
    """Save only trainable weights."""
    state = {k: v.cpu() for k, v in model.state_dict().items()
             if not k.startswith("qwen.")}
    torch.save(state, path)
    logger.info(f"  Saved checkpoint: {path}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 5: Two-phase QA overfit on ablation data (H200, ~2-3 hr)")

    parser.add_argument("--data_path", type=str, default=None,
                        help="QA data path (default: auto-detect)")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="Fraction of data for validation (default: 0.1)")
    parser.add_argument("--max_doc_tokens", type=int, default=1024,
                        help="Max document tokens (default: 1024)")
    parser.add_argument("--segment_len", type=int, default=128,
                        help="NTP continuation segment length (default: 128)")
    parser.add_argument("--num_queries", type=int, default=64,
                        help="Number of Perceiver queries (default: 64)")

    parser.add_argument("--ntp_steps", type=int, default=10000,
                        help="Phase 1 NTP training steps (default: 10000)")
    parser.add_argument("--qa_steps", type=int, default=10000,
                        help="Phase 2 QA training steps (default: 10000)")
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
    parser.add_argument("--ppl_gate", type=float, default=40.0,
                        help="Phase 1 val PPL gate (default: 40)")

    parser.add_argument("--use_teacher", action="store_true",
                        help="Enable teacher distillation in Phase 2")

    parser.add_argument("--log_every", type=int, default=50,
                        help="Log every N steps (default: 50)")
    parser.add_argument("--eval_every", type=int, default=500,
                        help="Evaluate every N steps (default: 500)")
    parser.add_argument("--save_every", type=int, default=5000,
                        help="Save every N steps (default: 5000)")

    parser.add_argument("--no_bf16", action="store_true",
                        help="Disable bf16 mixed precision")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader workers (default: 4 on CUDA)")

    parser.add_argument("--output_dir", type=str,
                        default="outputs/overfit_step5")
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

    use_bf16 = (device.type == "cuda" and not args.no_bf16
                and torch.cuda.is_bf16_supported())

    logger.info(f"Device: {device}")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.0f} GB)")
    logger.info(f"bf16: {'ON' if use_bf16 else 'OFF'}")
    logger.info(f"Data: {args.data_path}")
    logger.info(f"Phase 1 (NTP): {args.ntp_steps} steps  |  Phase 2 (QA): {args.qa_steps} steps")
    logger.info(f"LR: {args.lr}  Batch: {args.batch_size}x{args.grad_accum} "
                f"(effective {args.batch_size * args.grad_accum})")
    logger.info(f"PPL gate: {args.ppl_gate}  Teacher: {'ON' if args.use_teacher else 'OFF'}")
    logger.info(f"Doc tokens: {args.max_doc_tokens}  Segment len: {args.segment_len}")
    logger.info(f"Queries: {args.num_queries}")

    # Tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)

    if args.num_workers is not None:
        num_workers = args.num_workers
    else:
        num_workers = 4 if device.type == "cuda" else 0
    pin_memory = device.type == "cuda"

    # CSV log (shared across both phases)
    os.makedirs(args.output_dir, exist_ok=True)
    csv_path = os.path.join(args.output_dir, "learning_curve.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["phase", "step", "train_loss", "val_ntp_loss", "val_ntp_ppl",
                         "val_qa_loss", "lr", "elapsed_s"])

    t0 = time.time()

    # ── Phase 1: NTP on QA contexts ──────────────────────────────────

    logger.info("=" * 60)
    logger.info("  PHASE 1: NTP pretrain on QA context texts")
    logger.info("=" * 60)

    ntp_ds = QAContextNTPDataset(
        args.data_path, tokenizer,
        max_doc_tokens=args.max_doc_tokens,
        segment_len=args.segment_len,
    )
    n_total = len(ntp_ds)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val

    # Deterministic split
    indices = list(range(n_total))
    random.Random(42).shuffle(indices)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    ntp_train_ds = Subset(ntp_ds, train_indices)
    ntp_val_ds = Subset(ntp_ds, val_indices)
    logger.info(f"NTP dataset: {n_total} total -> {n_train} train / {n_val} val")

    ntp_train_loader = DataLoader(
        ntp_train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=num_workers, pin_memory=pin_memory,
    )
    ntp_val_loader = DataLoader(
        ntp_val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=num_workers, pin_memory=pin_memory,
    )

    ntp_config = build_ntp_config(args)
    logger.info("Loading Qwen3-0.6B + DeepCompressor (full model, identity proj)...")
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

    best_ppl = train_ntp_phase(
        model, ntp_train_loader, ntp_val_loader, ntp_config, device,
        csv_writer, csv_file, t0, use_bf16=use_bf16,
    )

    # PPL gate
    if best_ppl >= args.ppl_gate:
        csv_file.close()
        elapsed = time.time() - t0
        print("\n" + "=" * 60)
        print("  STEP 5: TWO-PHASE QA OVERFIT ON ABLATION DATA")
        print("=" * 60)
        print(f"  Best val PPL: {best_ppl:.1f} (gate: {args.ppl_gate})")
        print(f"\n  FAIL: Phase 1 val PPL >= {args.ppl_gate} — NTP did not converge.")
        print(f"  Phase 2 skipped. Try more steps or higher LR.")
        print(f"  Elapsed: {elapsed:.0f}s ({elapsed/3600:.1f}h)")
        print("=" * 60)
        sys.exit(1)

    logger.info(f"Phase 1 gate PASSED: best val PPL {best_ppl:.1f} < {args.ppl_gate}")

    # Free Phase 1 model
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ── Phase 2: QA fine-tune ─────────────────────────────────────────

    logger.info("=" * 60)
    logger.info("  PHASE 2: QA fine-tune from Phase 1 checkpoint")
    logger.info("=" * 60)

    # QA dataset with same train/val split indices
    qa_ds = QADataset(
        args.data_path, tokenizer,
        max_doc_tokens=args.max_doc_tokens,
    )
    qa_train_ds = Subset(qa_ds, train_indices)
    qa_val_ds = Subset(qa_ds, val_indices)
    logger.info(f"QA dataset: {n_total} total -> {n_train} train / {n_val} val")

    qa_train_loader = DataLoader(
        qa_train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collator, num_workers=num_workers, pin_memory=pin_memory,
    )
    qa_val_loader = DataLoader(
        qa_val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collator, num_workers=num_workers, pin_memory=pin_memory,
    )

    # Rebuild model with QA config (enables question conditioning)
    qa_config = build_qa_config(args)
    model_qa = DeepCompressor(qa_config)

    # Load Phase 1 best weights
    phase1_ckpt = os.path.join(args.output_dir, "phase1_best.pt")
    if not os.path.exists(phase1_ckpt):
        phase1_ckpt = os.path.join(args.output_dir, "phase1_final.pt")
    weights = torch.load(phase1_ckpt, map_location="cpu", weights_only=True)
    model_qa.load_state_dict(weights, strict=False)
    logger.info(f"Loaded Phase 1 checkpoint: {phase1_ckpt}")

    model_qa = model_qa.to(device)

    if qa_config.training.gradient_checkpointing:
        model_qa.qwen.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})

    # Teacher model (optional)
    teacher_model = None
    if args.use_teacher:
        from transformers import AutoModelForCausalLM
        logger.info("Loading teacher model (frozen Qwen3-0.6B for distillation)...")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            "models/Qwen3-0.6B",
            torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        )
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        for p in teacher_model.parameters():
            p.requires_grad = False
        logger.info("Teacher model loaded and frozen")

    best_val_loss = train_qa_phase(
        model_qa, qa_train_loader, qa_val_loader, qa_config, device,
        csv_writer, csv_file, t0, use_bf16=use_bf16,
        teacher_model=teacher_model,
    )

    # Generation evaluation on val set
    logger.info("Running generation evaluation on val set...")
    gen_metrics, gen_examples = evaluate_qa_generation(
        model_qa, qa_val_loader, tokenizer, device, use_bf16=use_bf16)
    logger.info(f"Val EM={gen_metrics['exact_match']:.1%}  "
                f"F1={gen_metrics['f1']:.1%}  "
                f"({gen_metrics['n_samples']} samples)")

    csv_file.close()
    elapsed = time.time() - t0

    # ── Summary ───────────────────────────────────────────────────────

    print("\n" + "=" * 60)
    print("  STEP 5: TWO-PHASE QA OVERFIT ON ABLATION DATA")
    print("=" * 60)
    print(f"  Device:       {device} {'(bf16)' if use_bf16 else '(fp32)'}")
    print(f"  Model:        identity proj, all stages, full layers")
    print(f"  Trainable:    {trainable:,} params")
    print(f"  Data:         {args.data_path} ({n_train} train / {n_val} val)")
    print(f"  Batch:        {args.batch_size}x{args.grad_accum} = "
          f"{args.batch_size * args.grad_accum} effective")
    print(f"  Teacher:      {'ON' if args.use_teacher else 'OFF'}")
    print(f"  Phase 1:      {args.ntp_steps} NTP steps -> best val PPL {best_ppl:.1f}")
    print(f"  Phase 2:      {args.qa_steps} QA steps -> best val loss {best_val_loss:.4f}")
    print(f"  Val EM:       {gen_metrics['exact_match']:.1%}")
    print(f"  Val F1:       {gen_metrics['f1']:.1%}")
    print(f"  Elapsed:      {elapsed:.0f}s ({elapsed/3600:.1f}h)")
    print(f"  Output:       {args.output_dir}/")

    # Show a few examples
    for i, ex in enumerate(gen_examples):
        print(f"\n  --- Example {i+1} ---")
        print(f"  Gold:   {ex['gold'][:200]}")
        print(f"  Pred:   {ex['pred'][:200]}")
        print(f"  EM={ex['em']:.0f}  F1={ex['f1']:.2f}")

    # Pass/fail verdict
    if gen_metrics["f1"] >= 0.3:
        print(f"\n  PASS: val F1 >= 30% — QA fine-tuning is learning to generate")
    elif best_val_loss < 3.0:
        print(f"\n  PARTIAL: val loss < 3.0 but F1 {gen_metrics['f1']:.1%} — may need more steps")
    else:
        print(f"\n  FAIL: val loss > 3.0, F1 {gen_metrics['f1']:.1%} — QA path not converging")
    print("=" * 60)


if __name__ == "__main__":
    main()
