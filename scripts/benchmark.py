#!/usr/bin/env python3
"""Unified benchmark comparison for Deep Compressor.

Compares the trained Deep Compressor pipeline against baselines:
  - direct_qwen:      Frozen Qwen reads full document + question (upper bound)
  - random_prefix:    Random Gaussian prefix (lower bound)
  - mean_pool_linear: Mean-pool doc hidden states + linear projection
  - mean_pool_mlp:    Mean-pool doc hidden states + 2-layer MLP projection
  - deep_compressor:  Full trained pipeline (requires checkpoint)

Usage:
  python scripts/benchmark.py \
      --config configs/benchmark.yaml \
      --checkpoint outputs/checkpoint-final/trainable_weights.pt \
      --eval_data data/qa_dev.json

  python scripts/benchmark.py \
      --config configs/benchmark.yaml \
      --eval_data data/qa_tiny_dev.json \
      --wandb --wandb_project dc-benchmark
"""

import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

from deep_compressor.config import DeepCompressorConfig
from deep_compressor.data import PaddingCollator, QADataset
from deep_compressor.eval import compute_exact_match, compute_f1
from deep_compressor.model import DeepCompressor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Baseline: Direct Qwen (upper bound) ─────────────────────────────

@torch.no_grad()
def eval_direct_qwen(qwen_model, tokenizer, eval_loader, device,
                     max_new_tokens=64):
    """Evaluate frozen Qwen reading the full document + question directly."""
    logger.info("Evaluating: direct_qwen (upper bound)")
    qwen_model.eval()

    all_em, all_f1 = [], []

    for batch in eval_loader:
        # Concatenate doc + question tokens
        doc_ids = batch["doc_input_ids"].to(device)
        doc_mask = batch["doc_attention_mask"].to(device)
        q_ids = batch["q_input_ids"].to(device)
        q_mask = batch["q_attention_mask"].to(device)

        input_ids = torch.cat([doc_ids, q_ids], dim=1)
        attention_mask = torch.cat([doc_mask, q_mask], dim=1)

        gen_ids = qwen_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        # Strip input portion
        gen_ids = gen_ids[:, input_ids.shape[1]:]

        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        golds = batch["answer_text"]

        for pred, gold in zip(preds, golds):
            all_em.append(compute_exact_match(pred, gold))
            all_f1.append(compute_f1(pred, gold))

    n = max(len(all_em), 1)
    return {"exact_match": sum(all_em) / n, "f1": sum(all_f1) / n,
            "n_samples": len(all_em)}


# ── Baseline: Random prefix (lower bound) ───────────────────────────

@torch.no_grad()
def eval_random_prefix(qwen_model, tokenizer, eval_loader, device,
                       num_queries, qwen_dim, max_new_tokens=64):
    """Evaluate with random Gaussian prefix embeddings."""
    logger.info("Evaluating: random_prefix (lower bound)")
    qwen_model.eval()
    embed_layer = qwen_model.model.embed_tokens

    all_em, all_f1 = [], []

    for batch in eval_loader:
        B = batch["q_input_ids"].shape[0]
        q_ids = batch["q_input_ids"].to(device)
        q_mask = batch["q_attention_mask"].to(device)

        # Random prefix
        prefix = torch.randn(B, num_queries, qwen_dim, device=device) * 0.02
        q_embeds = embed_layer(q_ids)
        inputs_embeds = torch.cat([prefix, q_embeds], dim=1)

        prefix_mask = torch.ones(B, num_queries, device=device, dtype=q_mask.dtype)
        attention_mask = torch.cat([prefix_mask, q_mask], dim=1)

        gen_ids = qwen_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen_ids = gen_ids[:, inputs_embeds.shape[1]:]

        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        golds = batch["answer_text"]

        for pred, gold in zip(preds, golds):
            all_em.append(compute_exact_match(pred, gold))
            all_f1.append(compute_f1(pred, gold))

    n = max(len(all_em), 1)
    return {"exact_match": sum(all_em) / n, "f1": sum(all_f1) / n,
            "n_samples": len(all_em)}


# ── Baseline: Mean-pool + projection ────────────────────────────────

class MeanPoolLinear(nn.Module):
    def __init__(self, dim, num_queries):
        super().__init__()
        self.base = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.proj = nn.Linear(dim, dim)

    def forward(self, doc_pooled):
        return self.base.unsqueeze(0) + self.proj(doc_pooled).unsqueeze(1)


class MeanPoolMLP(nn.Module):
    def __init__(self, dim, num_queries, hidden=None):
        super().__init__()
        hidden = hidden or dim
        self.base = nn.Parameter(torch.randn(num_queries, dim) * 0.02)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, doc_pooled):
        return self.base.unsqueeze(0) + self.mlp(doc_pooled).unsqueeze(1)


@torch.no_grad()
def eval_pool_baseline(tag, probe_cls, qwen_model, tokenizer,
                       eval_loader, device, num_queries, qwen_dim,
                       max_new_tokens=64):
    """Evaluate a simple mean-pool + projection baseline (untrained, random init)."""
    logger.info(f"Evaluating: {tag}")
    qwen_model.eval()
    embed_layer = qwen_model.model.embed_tokens
    probe = probe_cls(qwen_dim, num_queries).to(device)
    probe.eval()

    all_em, all_f1 = [], []

    for batch in eval_loader:
        doc_ids = batch["doc_input_ids"].to(device)
        doc_mask = batch["doc_attention_mask"].to(device)
        q_ids = batch["q_input_ids"].to(device)
        q_mask = batch["q_attention_mask"].to(device)

        # Get document hidden states
        out = qwen_model(input_ids=doc_ids, attention_mask=doc_mask,
                         output_hidden_states=True, use_cache=False)
        doc_hidden = out.hidden_states[-1]
        mask_f = doc_mask.unsqueeze(-1).float()
        doc_pooled = (doc_hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)

        prefix = probe(doc_pooled)
        q_embeds = embed_layer(q_ids)
        inputs_embeds = torch.cat([prefix, q_embeds], dim=1)

        B = q_ids.shape[0]
        prefix_mask = torch.ones(B, num_queries, device=device, dtype=q_mask.dtype)
        attention_mask = torch.cat([prefix_mask, q_mask], dim=1)

        gen_ids = qwen_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen_ids = gen_ids[:, inputs_embeds.shape[1]:]

        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        golds = batch["answer_text"]

        for pred, gold in zip(preds, golds):
            all_em.append(compute_exact_match(pred, gold))
            all_f1.append(compute_f1(pred, gold))

    n = max(len(all_em), 1)
    return {"exact_match": sum(all_em) / n, "f1": sum(all_f1) / n,
            "n_samples": len(all_em)}


# ── Deep Compressor evaluation ───────────────────────────────────────

@torch.no_grad()
def eval_deep_compressor(model, tokenizer, eval_loader, device,
                         max_new_tokens=64):
    """Evaluate the full Deep Compressor pipeline."""
    logger.info("Evaluating: deep_compressor")
    model.eval()

    all_em, all_f1 = [], []

    for batch in eval_loader:
        doc_ids = batch["doc_input_ids"].to(device)
        doc_mask = batch["doc_attention_mask"].to(device)
        q_ids = batch["q_input_ids"].to(device)
        q_mask = batch["q_attention_mask"].to(device)

        byte_array = model.encode_document(doc_ids, doc_mask)
        queries = model.encode_question(q_ids, q_mask)
        latent = model.compress(queries, byte_array, byte_mask=doc_mask)
        prefix = model.up_mlp(latent)

        gen_ids = model.generate_answer(
            prefix, q_ids, q_mask,
            tokenizer=tokenizer, max_new_tokens=max_new_tokens)

        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        golds = batch["answer_text"]

        for pred, gold in zip(preds, golds):
            all_em.append(compute_exact_match(pred, gold))
            all_f1.append(compute_f1(pred, gold))

    n = max(len(all_em), 1)
    return {"exact_match": sum(all_em) / n, "f1": sum(all_f1) / n,
            "n_samples": len(all_em)}


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark comparison for Deep Compressor")
    parser.add_argument("--config", type=str,
                        default="configs/benchmark.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained weights (.pt) for deep_compressor")
    parser.add_argument("--eval_data", type=str, required=True,
                        help="Path to QA eval data (JSON)")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max eval samples (0 = all)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated methods to run (default: all)")
    # wandb
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dc-benchmark")
    parser.add_argument("--wandb_entity", type=str, default=None)
    args = parser.parse_args()

    device = _device()
    logger.info(f"Device: {device}")

    # Load config
    config = DeepCompressorConfig.from_yaml(args.config)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.qwen.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load eval data
    eval_ds = QADataset(args.eval_data, tokenizer,
                        max_doc_tokens=config.qwen.max_doc_tokens,
                        max_question_tokens=config.qwen.max_question_tokens,
                        max_answer_tokens=config.qwen.max_answer_tokens)
    if args.max_samples > 0:
        eval_ds = Subset(eval_ds, list(range(min(args.max_samples, len(eval_ds)))))

    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
    eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                             shuffle=False, collate_fn=collator)
    logger.info(f"Eval samples: {len(eval_ds)}")

    # Load Qwen model (shared across methods)
    logger.info("Loading Qwen model...")
    qwen_model = AutoModelForCausalLM.from_pretrained(
        config.qwen.model_name_or_path, torch_dtype=torch.float32)
    qwen_model.to(device)
    qwen_model.eval()
    for p in qwen_model.parameters():
        p.requires_grad = False

    num_queries = config.effective_num_queries
    qwen_dim = config.qwen.hidden_size

    # Determine which methods to run
    all_methods = ["direct_qwen", "random_prefix", "mean_pool_linear",
                   "mean_pool_mlp", "deep_compressor"]
    if args.methods:
        methods = [m.strip() for m in args.methods.split(",")]
    else:
        methods = all_methods

    # Skip deep_compressor if no checkpoint provided
    if "deep_compressor" in methods and args.checkpoint is None:
        logger.warning("No --checkpoint provided, skipping deep_compressor")
        methods.remove("deep_compressor")

    # wandb init
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name="benchmark",
                config={"methods": methods, "n_samples": len(eval_ds)},
            )
        except ImportError:
            logger.warning("wandb not installed")

    # Run benchmarks
    results = {}

    if "direct_qwen" in methods:
        results["direct_qwen"] = eval_direct_qwen(
            qwen_model, tokenizer, eval_loader, device,
            args.max_new_tokens)

    if "random_prefix" in methods:
        results["random_prefix"] = eval_random_prefix(
            qwen_model, tokenizer, eval_loader, device,
            num_queries, qwen_dim, args.max_new_tokens)

    if "mean_pool_linear" in methods:
        results["mean_pool_linear"] = eval_pool_baseline(
            "mean_pool_linear", MeanPoolLinear,
            qwen_model, tokenizer, eval_loader, device,
            num_queries, qwen_dim, args.max_new_tokens)

    if "mean_pool_mlp" in methods:
        results["mean_pool_mlp"] = eval_pool_baseline(
            "mean_pool_mlp", MeanPoolMLP,
            qwen_model, tokenizer, eval_loader, device,
            num_queries, qwen_dim, args.max_new_tokens)

    if "deep_compressor" in methods:
        dc_model = DeepCompressor(config, qwen_model=qwen_model)
        weights = torch.load(args.checkpoint, map_location="cpu",
                             weights_only=True)
        dc_model.load_state_dict(weights, strict=False)
        dc_model.to(device)
        results["deep_compressor"] = eval_deep_compressor(
            dc_model, tokenizer, eval_loader, device,
            args.max_new_tokens)
        del dc_model

    # Print comparison table
    print("\n" + "=" * 70)
    print("  BENCHMARK RESULTS")
    print("=" * 70)
    print(f"\n  {'Method':<25}  {'EM':>8}  {'F1':>8}  {'Samples':>8}")
    print(f"  {'─' * 55}")

    for name, metrics in results.items():
        em = metrics["exact_match"]
        f1 = metrics["f1"]
        n = metrics["n_samples"]
        print(f"  {name:<25}  {em:>8.2%}  {f1:>8.4f}  {n:>8}")
    print(f"  {'─' * 55}")

    # Quality retention rate
    if "direct_qwen" in results and "deep_compressor" in results:
        upper_f1 = results["direct_qwen"]["f1"]
        dc_f1 = results["deep_compressor"]["f1"]
        if upper_f1 > 0:
            retention = dc_f1 / upper_f1
            print(f"\n  Quality retention (F1): {retention:.2%}")
            print(f"  Compression: {config.qwen.max_doc_tokens} tokens → "
                  f"{num_queries} prefix vectors")

    if "random_prefix" in results and "deep_compressor" in results:
        lower_f1 = results["random_prefix"]["f1"]
        dc_f1 = results["deep_compressor"]["f1"]
        upper_f1 = results.get("direct_qwen", {}).get("f1", dc_f1)
        range_val = upper_f1 - lower_f1
        if range_val > 0:
            normalized = (dc_f1 - lower_f1) / range_val
            print(f"  Normalized score (0=random, 1=direct): {normalized:.2%}")

    # Log to wandb
    if wandb_run:
        for name, metrics in results.items():
            for k, v in metrics.items():
                if k != "n_samples":
                    wandb_run.log({f"{name}/{k}": v})
        wandb_run.finish()

    print()


if __name__ == "__main__":
    main()
