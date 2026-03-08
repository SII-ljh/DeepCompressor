#!/usr/bin/env python3
"""Analyze actual sequence lengths in QA training data."""

import json
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def analyze_qa_data(data_path: str, tokenizer_path: str = "models/Qwen3-0.6B",
                    max_samples: int = 10000):
    """Analyze sequence lengths in QA JSON data."""

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    print(f"Loading QA data from {data_path}...")
    with open(data_path) as f:
        data = json.load(f)

    total_samples = len(data)
    print(f"Total samples: {total_samples:,}")

    if max_samples > 0 and total_samples > max_samples:
        print(f"Analyzing first {max_samples:,} samples for speed\n")
        data = data[:max_samples]
    else:
        print(f"Analyzing all {total_samples:,} samples\n")

    doc_lengths = []
    question_lengths = []
    answer_lengths = []
    total_lengths = []  # doc + question + answer

    for i, item in enumerate(data):
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i:,} samples...", end="\r")

        # Tokenize like QADataset does
        doc_tokens = tokenizer(item["context"], truncation=True, max_length=8192,
                              return_tensors="pt", padding=False)
        q_tokens = tokenizer(item["question"], truncation=True, max_length=256,
                           return_tensors="pt", padding=False)
        a_tokens = tokenizer(item["answer"], truncation=True, max_length=511,
                           return_tensors="pt", padding=False)

        doc_len = doc_tokens["input_ids"].shape[1]
        q_len = q_tokens["input_ids"].shape[1]
        a_len = a_tokens["input_ids"].shape[1] + 1  # +1 for EOS

        doc_lengths.append(doc_len)
        question_lengths.append(q_len)
        answer_lengths.append(a_len)
        total_lengths.append(doc_len + q_len + a_len)

    print(f"\n\nAnalyzed {len(doc_lengths):,} samples")
    print("=" * 70)

    print("\n📊 DOCUMENT LENGTHS (context):")
    print("-" * 70)
    arr = np.array(doc_lengths)
    print(f"  Min:      {arr.min():,} tokens")
    print(f"  Max:      {arr.max():,} tokens")
    print(f"  Mean:     {arr.mean():.1f} tokens")
    print(f"  Median:   {np.median(arr):.0f} tokens")
    print(f"  P95:      {np.percentile(arr, 95):.0f} tokens")
    print(f"  P99:      {np.percentile(arr, 99):.0f} tokens")

    print("\n📊 QUESTION LENGTHS:")
    print("-" * 70)
    arr_q = np.array(question_lengths)
    print(f"  Min:      {arr_q.min():,} tokens")
    print(f"  Max:      {arr_q.max():,} tokens")
    print(f"  Mean:     {arr_q.mean():.1f} tokens")
    print(f"  Median:   {np.median(arr_q):.0f} tokens")
    print(f"  P95:      {np.percentile(arr_q, 95):.0f} tokens")
    print(f"  P99:      {np.percentile(arr_q, 99):.0f} tokens")

    print("\n📊 ANSWER LENGTHS:")
    print("-" * 70)
    arr_a = np.array(answer_lengths)
    print(f"  Min:      {arr_a.min():,} tokens")
    print(f"  Max:      {arr_a.max():,} tokens")
    print(f"  Mean:     {arr_a.mean():.1f} tokens")
    print(f"  Median:   {np.median(arr_a):.0f} tokens")
    print(f"  P95:      {np.percentile(arr_a, 95):.0f} tokens")
    print(f"  P99:      {np.percentile(arr_a, 99):.0f} tokens")

    print("\n📊 TOTAL LENGTHS (doc + question + answer):")
    print("-" * 70)
    arr_t = np.array(total_lengths)
    print(f"  Min:      {arr_t.min():,} tokens")
    print(f"  Max:      {arr_t.max():,} tokens")
    print(f"  Mean:     {arr_t.mean():.1f} tokens")
    print(f"  Median:   {np.median(arr_t):.0f} tokens")
    print(f"  P95:      {np.percentile(arr_t, 95):.0f} tokens")
    print(f"  P99:      {np.percentile(arr_t, 99):.0f} tokens")

    # Check truncation at different thresholds
    print("\n⚠️  TRUNCATION ANALYSIS:")
    print("-" * 70)
    for threshold in [512, 1024, 2048, 4096, 8192]:
        truncated = sum(1 for x in doc_lengths if x >= threshold)
        pct = 100 * truncated / len(doc_lengths)
        print(f"  Documents >= {threshold:5d} tokens: {truncated:6,} ({pct:5.1f}%)")

    # Document length distribution
    print("\n📈 DOCUMENT LENGTH DISTRIBUTION:")
    print("-" * 70)
    bins = [0, 128, 256, 512, 1024, 2048, 4096, 8192, 20000]
    hist, _ = np.histogram(doc_lengths, bins=bins)
    for i in range(len(bins)-1):
        pct = 100 * hist[i] / len(doc_lengths)
        print(f"  {bins[i]:5d} - {bins[i+1]:5d}:  {hist[i]:7,} ({pct:5.1f}%)")

    # Memory impact simulation
    print("\n🔥 MEMORY IMPACT (batch_size=4, what PaddingCollator creates):")
    print("-" * 70)
    print("\nFirst 10 batches:")
    for batch_start in range(0, min(40, len(doc_lengths)), 4):
        batch_docs = doc_lengths[batch_start:batch_start+4]
        batch_qs = question_lengths[batch_start:batch_start+4]
        batch_as = answer_lengths[batch_start:batch_start+4]
        if len(batch_docs) < 4:
            continue

        doc_max = max(batch_docs)
        q_max = max(batch_qs)
        a_max = max(batch_as)

        # Total tokens after padding
        total_tokens = 4 * (doc_max + q_max + a_max)
        useful_tokens = sum(batch_docs) + sum(batch_qs) + sum(batch_as)
        waste_pct = 100 * (total_tokens - useful_tokens) / total_tokens

        print(f"  Batch {batch_start//4:3d}: doc_max={doc_max:4d}, q_max={q_max:3d}, "
              f"a_max={a_max:3d}  →  {total_tokens:5,} tokens ({waste_pct:.0f}% padding)")

    # Worst case
    worst_doc_max = max(max(doc_lengths[i:i+4]) for i in range(0, len(doc_lengths)-3, 4))
    worst_q_max = max(max(question_lengths[i:i+4]) for i in range(0, len(question_lengths)-3, 4))
    worst_a_max = max(max(answer_lengths[i:i+4]) for i in range(0, len(answer_lengths)-3, 4))
    worst_total = 4 * (worst_doc_max + worst_q_max + worst_a_max)

    print(f"\n  Worst batch: doc={worst_doc_max:,}, q={worst_q_max:,}, a={worst_a_max:,}")
    print(f"  → Total tokens: {worst_total:,}")
    print(f"  → Memory (bf16): {worst_total * 1024 * 2 / 1024 / 1024:.1f} MB (input only)")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR / "qa_train.json")

    if not Path(data_path).exists():
        print(f"ERROR: {data_path} not found")
        print("\nRun: python scripts/prepare_data.py")
        sys.exit(1)

    analyze_qa_data(data_path)
