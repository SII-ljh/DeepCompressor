#!/usr/bin/env python3
"""Comprehensive benchmark suite for Deep Compressor paper evaluation.

Features:
  - Internal QA evaluation (EM, F1, ROUGE-L)
  - External benchmarks (LongBench QA subsets)
  - Compression ratio vs performance sweep
  - NTP perplexity comparison
  - Inference latency measurement
  - Baselines: direct_qwen, random_prefix, truncated_context
  - Supports no-checkpoint mode (baselines only)

Usage:
  # Full evaluation
  python scripts/benchmark_suite.py \
      --config configs/benchmark.yaml \
      --checkpoint outputs/checkpoint-final/trainable_weights.pt \
      --eval_data data/qa_dev.json

  # Baselines only (no checkpoint needed)
  python scripts/benchmark_suite.py \
      --config configs/macbook_debug.yaml \
      --eval_data data/qa_tiny_dev.json \
      --benchmarks baselines --max_samples 5

  # LongBench only
  python scripts/benchmark_suite.py \
      --config configs/benchmark.yaml \
      --checkpoint outputs/checkpoint-final/trainable_weights.pt \
      --benchmarks longbench

  # Compression ratio sweep
  python scripts/benchmark_suite.py \
      --config configs/benchmark.yaml \
      --checkpoint outputs/checkpoint-final/trainable_weights.pt \
      --eval_data data/qa_dev.json \
      --benchmarks ratio_sweep --ratios 16,32,64,128,256

  # Specific LongBench subsets
  python scripts/benchmark_suite.py ... \
      --longbench_subsets multifieldqa_zh,hotpotqa
"""

import argparse
import copy
import json
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

from deep_compressor.benchmarks.dataset import BenchmarkQADataset
from deep_compressor.benchmarks.longbench import (
    LONGBENCH_QA_SUBSETS,
    load_from_local,
    load_longbench_subset,
)
from deep_compressor.config import DeepCompressorConfig
from deep_compressor.data import PaddingCollator, QADataset
from deep_compressor.eval import (
    LatencyTimer,
    compute_best_metric,
    compute_exact_match,
    compute_f1,
    compute_rouge_l,
    evaluate_qa_multi_ref,
)
from deep_compressor.model import DeepCompressor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Utilities ────────────────────────────────────────────────────────

def _device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ── Core functions ───────────────────────────────────────────────────

def build_model_for_ratio(base_config, qwen_model, checkpoint_path,
                          num_queries, device):
    """Build a DeepCompressor with a different num_queries, sharing the Qwen model.

    Handles query_init.base_queries shape mismatch:
      - num_queries < trained: slice first N rows
      - num_queries > trained: tile + small noise
      - equal: direct load
    """
    cfg = copy.deepcopy(base_config)
    cfg.ablation.override_num_queries = num_queries

    model = DeepCompressor(cfg, qwen_model=qwen_model)

    if checkpoint_path is not None:
        weights = torch.load(checkpoint_path, map_location="cpu",
                             weights_only=True)

        # Handle base_queries shape mismatch
        bq_key = "query_init.base_queries"
        if bq_key in weights:
            saved_bq = weights[bq_key]
            saved_nq = saved_bq.shape[0]

            if num_queries < saved_nq:
                weights[bq_key] = saved_bq[:num_queries]
            elif num_queries > saved_nq:
                repeats = (num_queries + saved_nq - 1) // saved_nq
                tiled = saved_bq.repeat(repeats, 1)[:num_queries]
                tiled = tiled + torch.randn_like(tiled) * 0.01
                weights[bq_key] = tiled

        model.load_state_dict(weights, strict=False)

    model.to(device)
    model.eval()
    return model


def evaluate_on_benchmark(model, tokenizer, records, config, device,
                          batch_size=4, max_new_tokens=64, max_samples=0):
    """Evaluate model on benchmark records, returning EM/F1/ROUGE-L.

    Args:
        model: DeepCompressor instance
        tokenizer: Qwen tokenizer
        records: list of {context, question, answer, answers} dicts
        config: DeepCompressorConfig
        device: torch device
        batch_size: evaluation batch size
        max_new_tokens: max generation length
        max_samples: limit samples (0 = all)

    Returns:
        dict with exact_match, f1, rouge_l, n_samples, avg_context_len
    """
    ds = BenchmarkQADataset(
        records, tokenizer,
        max_doc_tokens=config.qwen.max_doc_tokens,
        max_question_tokens=config.qwen.max_question_tokens,
        max_answer_tokens=config.qwen.max_answer_tokens)

    if max_samples > 0:
        ds = Subset(ds, list(range(min(max_samples, len(ds)))))

    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collator)

    metrics = evaluate_qa_multi_ref(model, loader, tokenizer, device,
                                    max_new_tokens=max_new_tokens)

    # Compute average context length
    total_ctx_len = sum(len(r.get("context", "")) for r in records)
    n_records = max_samples if 0 < max_samples < len(records) else len(records)
    metrics["avg_context_len"] = total_ctx_len / max(n_records, 1)

    return metrics


# ── Baselines ────────────────────────────────────────────────────────

@torch.no_grad()
def _eval_baseline_direct_qwen(qwen_model, tokenizer, eval_loader, device,
                                max_new_tokens=64):
    """Direct Qwen reads full document + question (upper bound)."""
    logger.info("Evaluating baseline: direct_qwen")
    qwen_model.eval()
    all_em, all_f1, all_rouge = [], [], []

    for batch in eval_loader:
        doc_ids = batch["doc_input_ids"].to(device)
        doc_mask = batch["doc_attention_mask"].to(device)
        q_ids = batch["q_input_ids"].to(device)
        q_mask = batch["q_attention_mask"].to(device)

        input_ids = torch.cat([doc_ids, q_ids], dim=1)
        attention_mask = torch.cat([doc_mask, q_mask], dim=1)

        gen_ids = qwen_model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id)
        gen_ids = gen_ids[:, input_ids.shape[1]:]

        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        if "all_answers" in batch:
            answers_list = batch["all_answers"]
        else:
            answers_list = [[g] for g in batch["answer_text"]]

        for pred, golds in zip(preds, answers_list):
            all_em.append(compute_best_metric(pred, golds, compute_exact_match))
            all_f1.append(compute_best_metric(pred, golds, compute_f1))
            all_rouge.append(compute_best_metric(pred, golds, compute_rouge_l))

    n = max(len(all_em), 1)
    return {"exact_match": sum(all_em) / n, "f1": sum(all_f1) / n,
            "rouge_l": sum(all_rouge) / n, "n_samples": len(all_em)}


@torch.no_grad()
def _eval_baseline_random_prefix(qwen_model, tokenizer, eval_loader, device,
                                  num_queries, qwen_dim, max_new_tokens=64):
    """Random Gaussian prefix (lower bound)."""
    logger.info("Evaluating baseline: random_prefix")
    qwen_model.eval()
    embed_layer = qwen_model.model.embed_tokens
    all_em, all_f1, all_rouge = [], [], []

    for batch in eval_loader:
        B = batch["q_input_ids"].shape[0]
        q_ids = batch["q_input_ids"].to(device)
        q_mask = batch["q_attention_mask"].to(device)

        prefix = torch.randn(B, num_queries, qwen_dim, device=device) * 0.02
        q_embeds = embed_layer(q_ids)
        inputs_embeds = torch.cat([prefix, q_embeds], dim=1)
        prefix_mask = torch.ones(B, num_queries, device=device, dtype=q_mask.dtype)
        attention_mask = torch.cat([prefix_mask, q_mask], dim=1)

        gen_ids = qwen_model.generate(
            inputs_embeds=inputs_embeds, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id)
        gen_ids = gen_ids[:, inputs_embeds.shape[1]:]

        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        if "all_answers" in batch:
            answers_list = batch["all_answers"]
        else:
            answers_list = [[g] for g in batch["answer_text"]]

        for pred, golds in zip(preds, answers_list):
            all_em.append(compute_best_metric(pred, golds, compute_exact_match))
            all_f1.append(compute_best_metric(pred, golds, compute_f1))
            all_rouge.append(compute_best_metric(pred, golds, compute_rouge_l))

    n = max(len(all_em), 1)
    return {"exact_match": sum(all_em) / n, "f1": sum(all_f1) / n,
            "rouge_l": sum(all_rouge) / n, "n_samples": len(all_em)}


@torch.no_grad()
def _eval_baseline_truncated(qwen_model, tokenizer, eval_loader, device,
                              num_queries, max_new_tokens=64):
    """Truncated context: only read first num_queries tokens of the document."""
    logger.info("Evaluating baseline: truncated_context")
    qwen_model.eval()
    all_em, all_f1, all_rouge = [], [], []

    for batch in eval_loader:
        doc_ids = batch["doc_input_ids"].to(device)[:, :num_queries]
        doc_mask = batch["doc_attention_mask"].to(device)[:, :num_queries]
        q_ids = batch["q_input_ids"].to(device)
        q_mask = batch["q_attention_mask"].to(device)

        input_ids = torch.cat([doc_ids, q_ids], dim=1)
        attention_mask = torch.cat([doc_mask, q_mask], dim=1)

        gen_ids = qwen_model.generate(
            input_ids=input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id)
        gen_ids = gen_ids[:, input_ids.shape[1]:]

        preds = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        if "all_answers" in batch:
            answers_list = batch["all_answers"]
        else:
            answers_list = [[g] for g in batch["answer_text"]]

        for pred, golds in zip(preds, answers_list):
            all_em.append(compute_best_metric(pred, golds, compute_exact_match))
            all_f1.append(compute_best_metric(pred, golds, compute_f1))
            all_rouge.append(compute_best_metric(pred, golds, compute_rouge_l))

    n = max(len(all_em), 1)
    return {"exact_match": sum(all_em) / n, "f1": sum(all_f1) / n,
            "rouge_l": sum(all_rouge) / n, "n_samples": len(all_em)}


def run_baselines(qwen_model, tokenizer, eval_loader, device, config,
                  max_new_tokens=64):
    """Run all baseline evaluations.

    Returns:
        dict mapping baseline name to metrics dict
    """
    num_queries = config.effective_num_queries
    qwen_dim = config.qwen.hidden_size

    results = {}
    results["direct_qwen"] = _eval_baseline_direct_qwen(
        qwen_model, tokenizer, eval_loader, device, max_new_tokens)
    results["random_prefix"] = _eval_baseline_random_prefix(
        qwen_model, tokenizer, eval_loader, device,
        num_queries, qwen_dim, max_new_tokens)
    results["truncated_context"] = _eval_baseline_truncated(
        qwen_model, tokenizer, eval_loader, device,
        num_queries, max_new_tokens)

    return results


# ── Compression ratio sweep ──────────────────────────────────────────

def run_compression_ratio_sweep(base_config, qwen_model, checkpoint_path,
                                tokenizer, eval_data_path, device,
                                ratios, batch_size=4, max_new_tokens=64,
                                max_samples=0):
    """Evaluate model at different compression ratios.

    Args:
        ratios: list of num_queries values to test

    Returns:
        dict mapping str(ratio) to {em, f1, rouge_l, compression_ratio}
    """
    max_doc_tokens = base_config.qwen.max_doc_tokens
    results = {}

    eval_ds = QADataset(eval_data_path, tokenizer,
                        max_doc_tokens=max_doc_tokens,
                        max_question_tokens=base_config.qwen.max_question_tokens,
                        max_answer_tokens=base_config.qwen.max_answer_tokens)
    if max_samples > 0:
        eval_ds = Subset(eval_ds, list(range(min(max_samples, len(eval_ds)))))

    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
    eval_loader = DataLoader(eval_ds, batch_size=batch_size,
                             shuffle=False, collate_fn=collator)

    for nq in ratios:
        logger.info(f"Ratio sweep: num_queries={nq}, "
                    f"compression_ratio={max_doc_tokens // nq}x")

        model = build_model_for_ratio(base_config, qwen_model,
                                      checkpoint_path, nq, device)

        metrics = evaluate_qa_multi_ref(model, eval_loader, tokenizer, device,
                                        max_new_tokens=max_new_tokens)

        results[str(nq)] = {
            "exact_match": metrics["exact_match"],
            "f1": metrics["f1"],
            "rouge_l": metrics["rouge_l"],
            "compression_ratio": max_doc_tokens // nq,
            "n_samples": metrics["n_samples"],
        }

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# ── NTP perplexity comparison ────────────────────────────────────────

@torch.no_grad()
def _compute_ntp_ppl(model_or_qwen, eval_loader, device, is_full_context=False):
    """Compute NTP perplexity.

    If is_full_context: uses raw Qwen model on full document tokens.
    Otherwise: uses DeepCompressor in NTP mode.
    """
    total_loss = 0.0
    total_tokens = 0

    for batch in eval_loader:
        if is_full_context:
            # Concatenate doc + segment, compute LM loss
            doc_ids = batch["doc_input_ids"].to(device)
            doc_mask = batch["doc_attention_mask"].to(device)
            seg_ids = batch["segment_ids"].to(device)
            seg_mask = batch["segment_attention_mask"].to(device)

            input_ids = torch.cat([doc_ids, seg_ids], dim=1)
            attention_mask = torch.cat([doc_mask, seg_mask], dim=1)

            # Labels: -100 for doc, segment ids for segment
            doc_labels = torch.full_like(doc_ids, -100)
            seg_labels = batch["segment_labels"].to(device)
            labels = torch.cat([doc_labels, seg_labels], dim=1)

            outputs = model_or_qwen(
                input_ids=input_ids, attention_mask=attention_mask,
                labels=labels, use_cache=False)
            loss = outputs.loss
            n_tokens = (seg_labels != -100).sum().item()
        else:
            losses = model_or_qwen(
                mode="ntp",
                doc_input_ids=batch["doc_input_ids"].to(device),
                doc_attention_mask=batch["doc_attention_mask"].to(device),
                segment_ids=batch["segment_ids"].to(device),
                segment_attention_mask=batch["segment_attention_mask"].to(device),
                segment_labels=batch["segment_labels"].to(device))
            loss = losses["total"]
            seg_labels = batch["segment_labels"]
            n_tokens = (seg_labels != -100).sum().item()

        total_loss += loss.item() * max(n_tokens, 1)
        total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return torch.exp(torch.tensor(avg_loss)).item()


def run_ntp_ppl_comparison(base_config, qwen_model, checkpoint_path,
                           tokenizer, ntp_data_path, device, ratios,
                           batch_size=4, max_samples=0):
    """Compare NTP perplexity: full context vs compressed at various ratios.

    Returns:
        {"full_context_ppl": float,
         str(ratio): {"ppl": float, "ppl_ratio": float}}
    """
    from deep_compressor.data import NTPDataset

    ntp_ds = NTPDataset(ntp_data_path, tokenizer,
                        max_doc_tokens=base_config.qwen.max_doc_tokens,
                        segment_len=base_config.training.ntp_segment_len)
    if max_samples > 0:
        ntp_ds = Subset(ntp_ds, list(range(min(max_samples, len(ntp_ds)))))

    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
    ntp_loader = DataLoader(ntp_ds, batch_size=batch_size,
                            shuffle=False, collate_fn=collator)

    # Full context PPL
    logger.info("Computing full-context NTP PPL...")
    full_ppl = _compute_ntp_ppl(qwen_model, ntp_loader, device,
                                is_full_context=True)
    logger.info(f"Full-context PPL: {full_ppl:.2f}")

    results = {"full_context_ppl": full_ppl}

    for nq in ratios:
        logger.info(f"NTP PPL: num_queries={nq}")
        model = build_model_for_ratio(base_config, qwen_model,
                                      checkpoint_path, nq, device)
        ppl = _compute_ntp_ppl(model, ntp_loader, device, is_full_context=False)
        results[str(nq)] = {
            "ppl": ppl,
            "ppl_ratio": ppl / max(full_ppl, 1e-8),
        }
        logger.info(f"  num_queries={nq}: PPL={ppl:.2f}, "
                    f"ratio={ppl / max(full_ppl, 1e-8):.2f}x")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return results


# ── Latency measurement ─────────────────────────────────────────────

@torch.no_grad()
def measure_latency(model, qwen_model, tokenizer, eval_data_path, config,
                    device, batch_size=1, max_new_tokens=64,
                    num_warmup=3, num_measure=10):
    """Measure inference latency for compressed vs full-context pipelines.

    Returns dict with per-stage timings and speedup factor.
    """
    eval_ds = QADataset(eval_data_path, tokenizer,
                        max_doc_tokens=config.qwen.max_doc_tokens,
                        max_question_tokens=config.qwen.max_question_tokens,
                        max_answer_tokens=config.qwen.max_answer_tokens)
    eval_ds = Subset(eval_ds, list(range(min(num_warmup + num_measure,
                                             len(eval_ds)))))

    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(eval_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=collator)

    batches = list(loader)
    model.eval()
    qwen_model.eval()

    # Warm up
    for batch in batches[:num_warmup]:
        doc_ids = batch["doc_input_ids"].to(device)
        doc_mask = batch["doc_attention_mask"].to(device)
        q_ids = batch["q_input_ids"].to(device)
        q_mask = batch["q_attention_mask"].to(device)
        _ = model.encode_document(doc_ids, doc_mask)

    # Measure compressed pipeline
    encode_times, compress_times, generate_times = [], [], []
    total_compressed_times = []

    measure_batches = batches[num_warmup:num_warmup + num_measure]
    if not measure_batches:
        measure_batches = batches[:num_measure]

    for batch in measure_batches:
        doc_ids = batch["doc_input_ids"].to(device)
        doc_mask = batch["doc_attention_mask"].to(device)
        q_ids = batch["q_input_ids"].to(device)
        q_mask = batch["q_attention_mask"].to(device)

        with LatencyTimer(device) as t_total:
            with LatencyTimer(device) as t_enc:
                byte_array = model.encode_document(doc_ids, doc_mask)
                queries = model.encode_question(q_ids, q_mask)

            with LatencyTimer(device) as t_comp:
                latent = model.compress(queries, byte_array, byte_mask=doc_mask)
                prefix = model.up_mlp(latent)

            with LatencyTimer(device) as t_gen:
                _ = model.generate_answer(
                    prefix, q_ids, q_mask,
                    tokenizer=tokenizer, max_new_tokens=max_new_tokens)

        encode_times.append(t_enc.elapsed_ms)
        compress_times.append(t_comp.elapsed_ms)
        generate_times.append(t_gen.elapsed_ms)
        total_compressed_times.append(t_total.elapsed_ms)

    # Measure full-context pipeline
    full_context_times = []
    for batch in measure_batches:
        doc_ids = batch["doc_input_ids"].to(device)
        doc_mask = batch["doc_attention_mask"].to(device)
        q_ids = batch["q_input_ids"].to(device)
        q_mask = batch["q_attention_mask"].to(device)

        input_ids = torch.cat([doc_ids, q_ids], dim=1)
        attention_mask = torch.cat([doc_mask, q_mask], dim=1)

        with LatencyTimer(device) as t_full:
            _ = qwen_model.generate(
                input_ids=input_ids, attention_mask=attention_mask,
                max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id)

        full_context_times.append(t_full.elapsed_ms)

    def _mean(lst):
        return sum(lst) / max(len(lst), 1)

    avg_compressed = _mean(total_compressed_times)
    avg_full = _mean(full_context_times)

    return {
        "encode_ms": round(_mean(encode_times), 2),
        "compress_ms": round(_mean(compress_times), 2),
        "generate_ms": round(_mean(generate_times), 2),
        "total_ms": round(avg_compressed, 2),
        "full_context_ms": round(avg_full, 2),
        "speedup": round(avg_full / max(avg_compressed, 0.01), 2),
        "n_measured": len(measure_batches),
    }


# ── Output functions ─────────────────────────────────────────────────

def save_results(results, output_path):
    """Save full results dict to JSON."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")


def print_summary_tables(results):
    """Print formatted summary tables to terminal."""
    print("\n" + "=" * 75)
    print("  BENCHMARK SUITE RESULTS")
    print("=" * 75)

    # Baselines
    if "baselines" in results:
        print(f"\n  Baselines:")
        print(f"  {'Method':<25}  {'EM':>8}  {'F1':>8}  {'ROUGE-L':>8}")
        print(f"  {'─' * 55}")
        for name, m in results["baselines"].items():
            print(f"  {name:<25}  {m['exact_match']:>8.2%}  {m['f1']:>8.4f}"
                  f"  {m['rouge_l']:>8.4f}")

    # Internal evaluation
    if "internal" in results:
        m = results["internal"]
        print(f"\n  Internal QA:")
        print(f"  {'deep_compressor':<25}  {m['exact_match']:>8.2%}  "
              f"{m['f1']:>8.4f}  {m['rouge_l']:>8.4f}")

    # LongBench
    if "longbench" in results:
        print(f"\n  LongBench External Benchmarks:")
        print(f"  {'Subset':<25}  {'EM':>8}  {'F1':>8}  {'ROUGE-L':>8}  {'N':>6}")
        print(f"  {'─' * 60}")
        for name, m in results["longbench"].items():
            print(f"  {name:<25}  {m['exact_match']:>8.2%}  {m['f1']:>8.4f}"
                  f"  {m['rouge_l']:>8.4f}  {m['n_samples']:>6}")

    # Ratio sweep
    if "ratio_sweep" in results:
        print(f"\n  Compression Ratio Sweep:")
        print(f"  {'Queries':>8}  {'Ratio':>8}  {'EM':>8}  {'F1':>8}  {'ROUGE-L':>8}")
        print(f"  {'─' * 50}")
        for nq, m in sorted(results["ratio_sweep"].items(), key=lambda x: int(x[0])):
            print(f"  {nq:>8}  {m['compression_ratio']:>7}x  "
                  f"{m['exact_match']:>8.2%}  {m['f1']:>8.4f}  {m['rouge_l']:>8.4f}")

    # NTP PPL
    if "ntp_ppl" in results:
        print(f"\n  NTP Perplexity:")
        full_ppl = results["ntp_ppl"]["full_context_ppl"]
        print(f"  Full context: {full_ppl:.2f}")
        print(f"  {'Queries':>8}  {'PPL':>10}  {'PPL Ratio':>10}")
        print(f"  {'─' * 35}")
        for key, val in results["ntp_ppl"].items():
            if key == "full_context_ppl":
                continue
            print(f"  {key:>8}  {val['ppl']:>10.2f}  {val['ppl_ratio']:>9.2f}x")

    # Latency
    if "latency" in results:
        lat = results["latency"]
        print(f"\n  Latency (avg over {lat.get('n_measured', '?')} samples):")
        print(f"    Encode:       {lat['encode_ms']:>8.1f} ms")
        print(f"    Compress:     {lat['compress_ms']:>8.1f} ms")
        print(f"    Generate:     {lat['generate_ms']:>8.1f} ms")
        print(f"    Total:        {lat['total_ms']:>8.1f} ms")
        print(f"    Full context: {lat['full_context_ms']:>8.1f} ms")
        print(f"    Speedup:      {lat['speedup']:>7.2f}x")

    print("\n" + "=" * 75 + "\n")


def log_to_wandb(results, wandb_run):
    """Log results to wandb."""
    if wandb_run is None:
        return

    flat = {}
    for section, data in results.items():
        if section == "metadata":
            continue
        if isinstance(data, dict):
            for key, val in data.items():
                if isinstance(val, dict):
                    for k2, v2 in val.items():
                        if isinstance(v2, (int, float)):
                            flat[f"{section}/{key}/{k2}"] = v2
                elif isinstance(val, (int, float)):
                    flat[f"{section}/{key}"] = val

    wandb_run.log(flat)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive benchmark suite for Deep Compressor")
    parser.add_argument("--config", type=str, default="configs/benchmark.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained weights (.pt)")
    parser.add_argument("--eval_data", type=str, default=None,
                        help="Path to internal QA eval data (JSON)")
    parser.add_argument("--ntp_data", type=str, default=None,
                        help="Path to NTP eval data (JSONL) for PPL comparison")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Max samples per benchmark (0 = all)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=64)

    parser.add_argument("--benchmarks", type=str, default="all",
                        help="Comma-separated: all, baselines, internal, "
                             "longbench, ratio_sweep, ntp_ppl, latency")
    parser.add_argument("--ratios", type=str, default="16,32,64,128,256",
                        help="Comma-separated num_queries for ratio sweep")
    parser.add_argument("--longbench_subsets", type=str, default=None,
                        help="Comma-separated LongBench subsets "
                             "(default: all 4 core QA subsets)")
    parser.add_argument("--longbench_cache_dir", type=str, default=None)
    parser.add_argument("--longbench_local_dir", type=str, default=None,
                        help="Load LongBench from local JSONs instead of HF")

    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: outputs/benchmark_results.json)")

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="dc-benchmark")
    parser.add_argument("--wandb_entity", type=str, default=None)
    args = parser.parse_args()

    device = _device()
    logger.info(f"Device: {device}")

    config = DeepCompressorConfig.from_yaml(args.config)
    ratios = [int(r) for r in args.ratios.split(",")]

    # Determine which benchmarks to run
    if args.benchmarks == "all":
        benchmarks = {"baselines", "internal", "longbench",
                      "ratio_sweep", "ntp_ppl", "latency"}
    else:
        benchmarks = {b.strip() for b in args.benchmarks.split(",")}

    # Skip benchmarks that need checkpoint if none provided
    needs_checkpoint = {"internal", "longbench", "ratio_sweep",
                        "ntp_ppl", "latency"}
    if args.checkpoint is None:
        skipped = benchmarks & needs_checkpoint
        if skipped:
            logger.warning(f"No --checkpoint provided, skipping: {skipped}")
            benchmarks -= needs_checkpoint

    # Skip benchmarks that need eval_data
    if args.eval_data is None:
        for b in ["baselines", "internal", "ratio_sweep", "latency"]:
            if b in benchmarks:
                logger.warning(f"No --eval_data provided, skipping: {b}")
                benchmarks.discard(b)

    if args.ntp_data is None and "ntp_ppl" in benchmarks:
        logger.warning("No --ntp_data provided, skipping: ntp_ppl")
        benchmarks.discard("ntp_ppl")

    if not benchmarks:
        logger.error("No benchmarks to run. Provide --eval_data and/or --checkpoint.")
        return

    logger.info(f"Benchmarks to run: {benchmarks}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.qwen.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load Qwen model (shared)
    logger.info("Loading Qwen model...")
    qwen_model = AutoModelForCausalLM.from_pretrained(
        config.qwen.model_name_or_path, torch_dtype=torch.float32)
    qwen_model.to(device)
    qwen_model.eval()
    for p in qwen_model.parameters():
        p.requires_grad = False

    # wandb
    wandb_run = None
    if args.wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project, entity=args.wandb_entity,
                name="benchmark_suite",
                config={"benchmarks": list(benchmarks),
                        "ratios": ratios,
                        "max_samples": args.max_samples})
        except ImportError:
            logger.warning("wandb not installed")

    # Build results dict
    results = {
        "metadata": {
            "config": args.config,
            "checkpoint": args.checkpoint,
            "timestamp": datetime.now().isoformat(),
            "model": config.qwen.model_name_or_path,
            "trained_num_queries": config.effective_num_queries,
            "max_doc_tokens": config.qwen.max_doc_tokens,
            "benchmarks": list(benchmarks),
        }
    }

    # Prepare internal eval loader (shared by baselines + internal)
    eval_loader = None
    if args.eval_data and (benchmarks & {"baselines", "internal", "latency"}):
        eval_ds = QADataset(args.eval_data, tokenizer,
                            max_doc_tokens=config.qwen.max_doc_tokens,
                            max_question_tokens=config.qwen.max_question_tokens,
                            max_answer_tokens=config.qwen.max_answer_tokens)
        if args.max_samples > 0:
            eval_ds = Subset(eval_ds,
                             list(range(min(args.max_samples, len(eval_ds)))))
        collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
        eval_loader = DataLoader(eval_ds, batch_size=args.batch_size,
                                 shuffle=False, collate_fn=collator)
        logger.info(f"Internal eval samples: {len(eval_ds)}")

    # ── Run benchmarks ───────────────────────────────────────────────

    if "baselines" in benchmarks:
        results["baselines"] = run_baselines(
            qwen_model, tokenizer, eval_loader, device, config,
            max_new_tokens=args.max_new_tokens)

    if "internal" in benchmarks:
        logger.info("Evaluating: deep_compressor (internal)")
        dc_model = build_model_for_ratio(
            config, qwen_model, args.checkpoint,
            config.effective_num_queries, device)
        metrics = evaluate_qa_multi_ref(
            dc_model, eval_loader, tokenizer, device,
            max_new_tokens=args.max_new_tokens)
        results["internal"] = metrics
        del dc_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "longbench" in benchmarks:
        lb_subsets = (args.longbench_subsets.split(",")
                      if args.longbench_subsets
                      else list(LONGBENCH_QA_SUBSETS.keys()))
        results["longbench"] = {}

        dc_model = build_model_for_ratio(
            config, qwen_model, args.checkpoint,
            config.effective_num_queries, device)

        for subset_name in lb_subsets:
            logger.info(f"Evaluating LongBench: {subset_name}")
            try:
                if args.longbench_local_dir:
                    local_path = os.path.join(args.longbench_local_dir,
                                              f"{subset_name}.json")
                    records = load_from_local(local_path)
                else:
                    records = load_longbench_subset(
                        subset_name, cache_dir=args.longbench_cache_dir)

                metrics = evaluate_on_benchmark(
                    dc_model, tokenizer, records, config, device,
                    batch_size=args.batch_size,
                    max_new_tokens=args.max_new_tokens,
                    max_samples=args.max_samples)
                results["longbench"][subset_name] = metrics
            except Exception as e:
                logger.error(f"Failed to evaluate {subset_name}: {e}")
                results["longbench"][subset_name] = {"error": str(e)}

        del dc_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if "ratio_sweep" in benchmarks:
        results["ratio_sweep"] = run_compression_ratio_sweep(
            config, qwen_model, args.checkpoint, tokenizer,
            args.eval_data, device, ratios,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
            max_samples=args.max_samples)

    if "ntp_ppl" in benchmarks:
        results["ntp_ppl"] = run_ntp_ppl_comparison(
            config, qwen_model, args.checkpoint, tokenizer,
            args.ntp_data, device, ratios,
            batch_size=args.batch_size,
            max_samples=args.max_samples)

    if "latency" in benchmarks:
        dc_model = build_model_for_ratio(
            config, qwen_model, args.checkpoint,
            config.effective_num_queries, device)
        results["latency"] = measure_latency(
            dc_model, qwen_model, tokenizer, args.eval_data, config, device,
            batch_size=1, max_new_tokens=args.max_new_tokens)
        del dc_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Output ───────────────────────────────────────────────────────

    print_summary_tables(results)

    output_path = args.output or os.path.join(
        config.training.output_dir, "benchmark_results.json")
    save_results(results, output_path)

    if wandb_run:
        log_to_wandb(results, wandb_run)
        wandb_run.finish()


if __name__ == "__main__":
    main()
