#!/usr/bin/env python3
"""Analyze actual sequence lengths in NTP training data."""

import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def analyze_ntp_data(data_path: str, tokenizer_path: str = "models/Qwen3-0.6B",
                     max_samples: int = 10000, max_doc_tokens: int = 8192):
    """Analyze sequence lengths in NTP JSONL data."""

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    print(f"Analyzing {data_path}...")
    print(f"Will sample up to {max_samples:,} entries for speed\n")

    doc_lengths = []
    total_lengths = []

    with open(data_path) as f:
        for i, line in enumerate(f):
            if i >= max_samples:
                break
            if i % 1000 == 0 and i > 0:
                print(f"  Processed {i:,} samples...", end="\r")

            data = json.loads(line)
            text = data["text"]

            # Tokenize like NTPDataset does
            tokens = tokenizer(text, truncation=True,
                             max_length=max_doc_tokens + 256,  # segment_len=256
                             return_tensors="pt", padding=False)
            total_len = tokens["input_ids"].shape[1]
            total_lengths.append(total_len)

            # Split point (simulate what NTPDataset does)
            if total_len <= 256 + 1:
                split_point = total_len // 2
            else:
                max_split = total_len - 256
                split_point = min(max_split, max_doc_tokens)

            doc_lengths.append(split_point)

    print(f"\n\nAnalyzed {len(doc_lengths):,} samples")
    print("=" * 60)

    print("\n📊 DOCUMENT LENGTHS (after split, what goes to encoder):")
    print("-" * 60)
    arr = np.array(doc_lengths)
    print(f"  Min:      {arr.min():,} tokens")
    print(f"  Max:      {arr.max():,} tokens")
    print(f"  Mean:     {arr.mean():.1f} tokens")
    print(f"  Median:   {np.median(arr):.0f} tokens")
    print(f"  P95:      {np.percentile(arr, 95):.0f} tokens")
    print(f"  P99:      {np.percentile(arr, 99):.0f} tokens")

    print("\n📊 TOTAL LENGTHS (before split, raw tokenized):")
    print("-" * 60)
    arr2 = np.array(total_lengths)
    print(f"  Min:      {arr2.min():,} tokens")
    print(f"  Max:      {arr2.max():,} tokens")
    print(f"  Mean:     {arr2.mean():.1f} tokens")
    print(f"  Median:   {np.median(arr2):.0f} tokens")
    print(f"  P95:      {np.percentile(arr2, 95):.0f} tokens")
    print(f"  P99:      {np.percentile(arr2, 99):.0f} tokens")

    # Check how many are actually truncated to max_doc_tokens
    at_max = sum(1 for x in doc_lengths if x >= max_doc_tokens)
    print(f"\n⚠️  Samples at max_doc_tokens ({max_doc_tokens}): {at_max:,} ({100*at_max/len(doc_lengths):.1f}%)")

    # Length distribution bins
    print("\n📈 DOCUMENT LENGTH DISTRIBUTION:")
    print("-" * 60)
    bins = [0, 512, 1024, 2048, 4096, 6144, 8192, 10000]
    hist, _ = np.histogram(doc_lengths, bins=bins)
    for i in range(len(bins)-1):
        pct = 100 * hist[i] / len(doc_lengths)
        print(f"  {bins[i]:5d} - {bins[i+1]:5d}:  {hist[i]:6,} ({pct:5.1f}%)")

    print("\n" + "=" * 60)

    # After collation, what's the actual batch max?
    print("\n🔥 MEMORY IMPACT (batch_size=4, what PaddingCollator will create):")
    print("-" * 60)
    # Simulate batches
    for batch_start in range(0, min(100, len(doc_lengths)), 4):
        batch = doc_lengths[batch_start:batch_start+4]
        if len(batch) < 4:
            continue
        batch_max = max(batch)
        print(f"  Batch {batch_start//4:3d}: max_len = {batch_max:5,} tokens  "
              f"(padding {4*batch_max - sum(batch):5,} tokens wasted)")
        if batch_start >= 20:  # Show first 5 batches
            break

    worst_batch_max = max(max(doc_lengths[i:i+4]) for i in range(0, len(doc_lengths), 4) if i+4 <= len(doc_lengths))
    print(f"\n  Worst batch max_len: {worst_batch_max:,} tokens")
    print(f"  → Per-batch memory:  {worst_batch_max * 4 / 1024:.1f}K tokens × hidden_dim")


if __name__ == "__main__":
    data_path = sys.argv[1] if len(sys.argv) > 1 else str(DATA_DIR / "ntp_train.jsonl")

    if not Path(data_path).exists():
        print(f"ERROR: {data_path} not found")
        print("\nRun: python scripts/prepare_data.py")
        sys.exit(1)

    analyze_ntp_data(data_path)
