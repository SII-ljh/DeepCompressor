#!/usr/bin/env python3
"""Fast QA dataset filtering using cached length index.

Strategy:
1. First run: build length index (qa_train_lengths.json) - ONE TIME COST
2. Subsequent runs: use cached index to filter instantly - NO tokenization

Usage:
  # First time (builds index)
  python scripts/filter_qa_by_length.py --max_tokens 512

  # Second time (instant, reuses index)
  python scripts/filter_qa_by_length.py --min_tokens 512 --max_tokens 2048
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def build_length_index(data_path: str, tokenizer_path: str):
    """Build and cache document length index.

    Returns: dict mapping sample index -> doc_length
    """
    index_path = data_path.replace(".json", "_lengths.json")

    # Check if index already exists
    if Path(index_path).exists():
        print(f"✓ Found cached length index: {index_path}")
        with open(index_path) as f:
            return json.load(f)

    print(f"Building length index for {data_path}...")
    print("(This is a ONE-TIME operation, result will be cached)")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    with open(data_path) as f:
        data = json.load(f)

    total = len(data)
    lengths = {}

    import time
    start = time.time()

    for i, item in enumerate(data):
        if i % 100 == 0 and i > 0:
            elapsed = time.time() - start
            rate = i / elapsed
            eta = (total - i) / rate if rate > 0 else 0
            print(f"  {i:,}/{total:,} ({100*i/total:.1f}%)  "
                  f"{rate:.0f} samples/s  ETA: {eta/60:.1f}min", end="\r")

        tokens = tokenizer.encode(item["context"], truncation=False)
        lengths[str(i)] = len(tokens)

    print(f"\n✓ Indexed {total:,} samples in {time.time()-start:.1f}s")

    # Save index
    print(f"Saving index to {index_path}...")
    with open(index_path, "w") as f:
        json.dump(lengths, f)

    print(f"✓ Index cached. Future runs will be instant!\n")
    return lengths


def filter_with_index(data_path: str, output_path: str,
                     lengths: dict, min_tokens: int, max_tokens: int):
    """Filter dataset using pre-computed length index (instant)."""

    print(f"Loading data from {data_path}...")
    with open(data_path) as f:
        data = json.load(f)

    total = len(data)
    print(f"Total samples: {total:,}")

    if min_tokens > 0:
        print(f"Filtering to {min_tokens} <= doc_tokens <= {max_tokens}...\n")
    else:
        print(f"Filtering to doc_tokens <= {max_tokens}...\n")

    # Fast filtering using index (no tokenization!)
    filtered = []
    for i, item in enumerate(data):
        doc_len = lengths[str(i)]
        if min_tokens <= doc_len <= max_tokens:
            filtered.append(item)

    print(f"Filtered results:")
    print(f"  Input:  {total:,} samples")
    print(f"  Output: {len(filtered):,} samples ({100*len(filtered)/total:.1f}%)")
    print(f"  Dropped: {total - len(filtered):,} samples\n")

    print(f"Saving to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(filtered, f, ensure_ascii=False)

    print(f"✓ Done! Saved {len(filtered):,} samples")


def main():
    parser = argparse.ArgumentParser(
        description="Fast QA filtering using cached length index"
    )
    parser.add_argument("--min_tokens", type=int, default=0)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--tokenizer", type=str, default="models/Qwen3-0.6B")
    parser.add_argument("--input_train", type=str, default=str(DATA_DIR / "qa_train.json"))
    parser.add_argument("--input_dev", type=str, default=str(DATA_DIR / "qa_dev.json"))
    args = parser.parse_args()

    # Build/load length index for train (ONE TIME COST)
    train_lengths = build_length_index(args.input_train, args.tokenizer)

    print("=" * 70)

    # Build/load length index for dev
    dev_lengths = build_length_index(args.input_dev, args.tokenizer)

    print("=" * 70 + "\n")

    # Generate output filename
    if args.min_tokens > 0:
        suffix = f"{args.min_tokens}_{args.max_tokens}"
    else:
        suffix = f"{args.max_tokens}"

    # Filter train (INSTANT with index)
    output_train = DATA_DIR / f"qa_train_filtered_{suffix}.json"
    filter_with_index(args.input_train, str(output_train),
                     train_lengths, args.min_tokens, args.max_tokens)

    print("\n" + "=" * 70 + "\n")

    # Filter dev
    output_dev = DATA_DIR / f"qa_dev_filtered_{suffix}.json"
    filter_with_index(args.input_dev, str(output_dev),
                     dev_lengths, args.min_tokens, args.max_tokens)

    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  Train: {output_train}")
    print(f"  Dev:   {output_dev}")
    if args.min_tokens > 0:
        print(f"\nFiltered to {args.min_tokens} <= doc_tokens <= {args.max_tokens}")
    else:
        print(f"\nFiltered to doc_tokens <= {args.max_tokens}")


if __name__ == "__main__":
    main()
