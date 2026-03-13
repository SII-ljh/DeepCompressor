#!/usr/bin/env python3
"""Analyze token length distribution of QA datasets.

Uses multiprocessing to tokenize in parallel across all available CPU cores
(or across multiple GPUs for the per-GPU memory estimation).

Usage:
    # Basic — analyze the training set
    python scripts/analyze_data_lengths.py --data data/qa_large_train.json

    # Full — training + eval, with per-GPU batch size recommendation
    python scripts/analyze_data_lengths.py \
        --data data/qa_large_train.json \
        --eval_data data/qa_large_dev.json \
        --model models/Qwen3-0.6B \
        --max_doc 4096 --max_q 256 --max_a 512

    # Fast — only sample 10k entries (for very large datasets)
    python scripts/analyze_data_lengths.py \
        --data data/qa_large_train.json --sample 10000

    # 8-GPU parallel tokenization (splits data across workers)
    python scripts/analyze_data_lengths.py \
        --data data/qa_large_train.json --num_workers 8
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np

# Add project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Tokenization worker (runs in subprocess) ─────────────────────────

def _tokenize_chunk(items, tokenizer_path, max_doc, max_q, max_a):
    """Tokenize a chunk of items. Runs in a worker process."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    doc_lens = []
    q_lens = []
    a_lens = []

    for item in items:
        doc = tokenizer(item["context"], truncation=True, max_length=max_doc,
                        return_tensors=None, padding=False)
        q = tokenizer(item["question"], truncation=True, max_length=max_q,
                      return_tensors=None, padding=False)
        a = tokenizer(item["answer"], truncation=True, max_length=max_a - 1,
                      return_tensors=None, padding=False)

        doc_lens.append(len(doc["input_ids"]))
        q_lens.append(len(q["input_ids"]))
        a_lens.append(len(a["input_ids"]) + 1)  # +1 for EOS

    return doc_lens, q_lens, a_lens


def _analyze_field(name, lengths, max_len):
    """Print statistics for a single field."""
    arr = np.array(lengths)
    p = np.percentile(arr, [25, 50, 75, 90, 95, 99])
    truncated = np.sum(arr >= max_len)
    trunc_pct = 100.0 * truncated / len(arr)

    print(f"\n  {name} (max_length={max_len}, n={len(arr):,})")
    print(f"  {'─' * 60}")
    print(f"  Min:    {arr.min():>8,}    Max:  {arr.max():>8,}    Mean: {arr.mean():>10.1f}")
    print(f"  P25:    {int(p[0]):>8,}    P50:  {int(p[1]):>8,}    P75:  {int(p[2]):>8,}")
    print(f"  P90:    {int(p[3]):>8,}    P95:  {int(p[4]):>8,}    P99:  {int(p[5]):>8,}")
    print(f"  Truncated (>={max_len}): {truncated:,} ({trunc_pct:.1f}%)")

    return {
        "name": name,
        "min": int(arr.min()), "max": int(arr.max()),
        "mean": float(arr.mean()), "std": float(arr.std()),
        "p25": int(p[0]), "p50": int(p[1]), "p75": int(p[2]),
        "p90": int(p[3]), "p95": int(p[4]), "p99": int(p[5]),
        "truncated": int(truncated), "truncated_pct": float(trunc_pct),
    }


def _print_histogram(name, lengths, max_len, bins=20):
    """Print a text histogram."""
    arr = np.array(lengths)
    edges = np.linspace(0, max_len, bins + 1)
    counts, _ = np.histogram(arr, bins=edges)
    max_count = max(counts) if max(counts) > 0 else 1
    bar_width = 40

    print(f"\n  {name} — Distribution Histogram")
    print(f"  {'─' * 60}")
    for i in range(bins):
        lo, hi = int(edges[i]), int(edges[i + 1])
        bar_len = int(bar_width * counts[i] / max_count)
        bar = "█" * bar_len
        pct = 100.0 * counts[i] / len(arr)
        print(f"  {lo:>6}-{hi:<6} │{bar:<{bar_width}} {counts[i]:>8,} ({pct:5.1f}%)")


def _print_batch_recommendation(doc_stats, q_stats, a_stats):
    """Print batch size recommendations based on data distribution."""
    print("\n" + "=" * 70)
    print("  BATCH SIZE RECOMMENDATIONS")
    print("=" * 70)

    # Memory is roughly proportional to total sequence length
    # Use different percentiles for conservative vs aggressive
    for label, percentile_key in [
        ("Conservative (P99)", "p99"),
        ("Balanced (P95)", "p95"),
        ("Aggressive (P90)", "p90"),
        ("Very aggressive (P75)", "p75"),
    ]:
        total = doc_stats[percentile_key] + q_stats[percentile_key] + a_stats[percentile_key]
        max_total = doc_stats["max"] + q_stats["max"] + a_stats["max"]
        ratio = total / max_total if max_total > 0 else 1.0
        print(f"\n  {label}:")
        print(f"    Probe lengths: doc={doc_stats[percentile_key]}, "
              f"q={q_stats[percentile_key]}, a={a_stats[percentile_key]}")
        print(f"    Total: {total:,} tokens  "
              f"({ratio:.1%} of worst-case {max_total:,})")
        print(f"    → If worst-case batch fits N samples, "
              f"this fits ~{1/ratio:.1f}x N samples")

    # Warn about long-tail OOM risk
    print(f"\n  ⚠  OOM risk at each level:")
    for label, pkey in [("P99", "p99"), ("P95", "p95"), ("P90", "p90"), ("P75", "p75")]:
        exceed_doc = 100.0 - float(pkey[1:])
        print(f"    {label}: ~{exceed_doc:.0f}% of batches may have ≥1 longer sample")


def analyze_dataset(data_path, tokenizer_path, max_doc, max_q, max_a,
                    num_workers, sample_size, label=""):
    """Analyze one dataset file."""
    print(f"\n{'=' * 70}")
    print(f"  DATASET: {data_path}  {label}")
    print(f"{'=' * 70}")

    t0 = time.time()
    with open(data_path) as f:
        data = json.load(f)
    print(f"  Loaded {len(data):,} samples in {time.time() - t0:.1f}s")

    if sample_size and sample_size < len(data):
        rng = np.random.default_rng(42)
        indices = rng.choice(len(data), size=sample_size, replace=False)
        data = [data[i] for i in indices]
        print(f"  Sampled {len(data):,} for analysis")

    # Split data into chunks for parallel processing
    effective_workers = min(num_workers, len(data))
    chunk_size = (len(data) + effective_workers - 1) // effective_workers
    chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

    print(f"  Tokenizing with {effective_workers} workers...")
    t0 = time.time()

    all_doc, all_q, all_a = [], [], []

    if effective_workers <= 1:
        # Single-process — avoid subprocess overhead
        doc, q, a = _tokenize_chunk(data, tokenizer_path, max_doc, max_q, max_a)
        all_doc, all_q, all_a = doc, q, a
    else:
        worker_fn = partial(_tokenize_chunk,
                            tokenizer_path=tokenizer_path,
                            max_doc=max_doc, max_q=max_q, max_a=max_a)
        with ProcessPoolExecutor(max_workers=effective_workers) as pool:
            futures = {pool.submit(worker_fn, chunk): i
                       for i, chunk in enumerate(chunks)}
            for fut in as_completed(futures):
                doc, q, a = fut.result()
                all_doc.extend(doc)
                all_q.extend(q)
                all_a.extend(a)

    elapsed = time.time() - t0
    rate = len(data) / elapsed
    print(f"  Tokenized {len(data):,} samples in {elapsed:.1f}s ({rate:,.0f} samples/s)")

    # Statistics
    doc_stats = _analyze_field("doc (context)", all_doc, max_doc)
    q_stats = _analyze_field("question", all_q, max_q)
    a_stats = _analyze_field("answer", all_a, max_a)

    # Total sequence length per sample
    totals = np.array(all_doc) + np.array(all_q) + np.array(all_a)
    max_total = max_doc + max_q + max_a
    total_stats = _analyze_field("total (doc+q+a)", totals.tolist(), max_total)

    # Histograms
    _print_histogram("doc (context)", all_doc, max_doc)
    _print_histogram("question", all_q, max_q)
    _print_histogram("answer", all_a, max_a)

    # Recommendations
    _print_batch_recommendation(doc_stats, q_stats, a_stats)

    return {
        "path": data_path,
        "n_samples": len(data),
        "doc": doc_stats, "question": q_stats, "answer": a_stats,
        "total": total_stats,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze token length distribution of QA datasets")
    parser.add_argument("--data", default="data/qa_large_train.json",
                        help="Path to training data (default: data/qa_large_train.json)")
    parser.add_argument("--eval_data", default="data/qa_large_dev.json",
                        help="Path to eval data (default: data/qa_large_dev.json)")
    parser.add_argument("--model", default="models/Qwen3-0.6B",
                        help="Tokenizer model path (default: models/Qwen3-0.6B)")
    parser.add_argument("--max_doc", type=int, default=4096,
                        help="Max doc tokens (default: 4096)")
    parser.add_argument("--max_q", type=int, default=256,
                        help="Max question tokens (default: 256)")
    parser.add_argument("--max_a", type=int, default=512,
                        help="Max answer tokens (default: 512)")
    parser.add_argument("--num_workers", type=int, default=20,
                        help="Parallel workers for tokenization (default: 20)")
    parser.add_argument("--sample", type=int, default=0,
                        help="Random sample size (0=all, default: 0)")
    parser.add_argument("--output", default="results/data_length_stats.json",
                        help="Save results to JSON file (default: results/data_length_stats.json)")
    args = parser.parse_args()

    results = {}

    results["train"] = analyze_dataset(
        args.data, args.model,
        args.max_doc, args.max_q, args.max_a,
        args.num_workers, args.sample or None,
        label="(train)",
    )

    if args.eval_data:
        results["eval"] = analyze_dataset(
            args.eval_data, args.model,
            args.max_doc, args.max_q, args.max_a,
            args.num_workers, args.sample or None,
            label="(eval)",
        )

    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n  Results saved to {args.output}")

    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
