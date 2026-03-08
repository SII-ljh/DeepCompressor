#!/usr/bin/env python3
"""Filter QA dataset by document length.

Creates filtered versions of qa_train.json and qa_dev.json containing only
samples with doc_len <= max_doc_tokens.

Usage:
  python scripts/filter_qa_by_length.py --max_tokens 512
  python scripts/filter_qa_by_length.py --max_tokens 2048
"""

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def filter_qa_data(input_path: str, output_path: str, tokenizer_path: str,
                   min_doc_tokens: int, max_doc_tokens: int):
    """Filter QA dataset to only include samples with min_doc_tokens <= doc_len <= max_doc_tokens."""

    print(f"Loading tokenizer from {tokenizer_path}...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    print(f"Loading QA data from {input_path}...")
    with open(input_path) as f:
        data = json.load(f)

    total = len(data)
    print(f"Total samples: {total:,}")
    if min_doc_tokens > 0:
        print(f"Filtering to {min_doc_tokens} <= doc_tokens <= {max_doc_tokens}...\n")
    else:
        print(f"Filtering to doc_tokens <= {max_doc_tokens}...\n")

    filtered = []
    for i, item in enumerate(data):
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i:,}/{total:,}...", end="\r")

        # Tokenize context
        tokens = tokenizer(item["context"], truncation=False,
                          return_tensors="pt", padding=False)
        doc_len = tokens["input_ids"].shape[1]

        if min_doc_tokens <= doc_len <= max_doc_tokens:
            filtered.append(item)

    print(f"\n\nFiltered results:")
    print(f"  Input:  {total:,} samples")
    print(f"  Output: {len(filtered):,} samples ({100*len(filtered)/total:.1f}%)")
    print(f"  Dropped: {total - len(filtered):,} samples\n")

    print(f"Saving to {output_path}...")
    with open(output_path, "w") as f:
        json.dump(filtered, f, ensure_ascii=False, indent=2)

    print(f"✓ Done! Saved {len(filtered):,} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Filter QA dataset by document length"
    )
    parser.add_argument(
        "--min_tokens", type=int, default=0,
        help="Minimum document tokens (default: 0)"
    )
    parser.add_argument(
        "--max_tokens", type=int, default=512,
        help="Maximum document tokens (default: 512)"
    )
    parser.add_argument(
        "--tokenizer", type=str, default="models/Qwen3-0.6B",
        help="Tokenizer path (default: models/Qwen3-0.6B)"
    )
    parser.add_argument(
        "--input_train", type=str, default=str(DATA_DIR / "qa_train.json"),
        help="Input QA train file"
    )
    parser.add_argument(
        "--input_dev", type=str, default=str(DATA_DIR / "qa_dev.json"),
        help="Input QA dev file"
    )
    args = parser.parse_args()

    # Generate output filename
    if args.min_tokens > 0:
        suffix = f"{args.min_tokens}_{args.max_tokens}"
    else:
        suffix = f"{args.max_tokens}"

    # Filter train
    output_train = DATA_DIR / f"qa_train_filtered_{suffix}.json"
    filter_qa_data(args.input_train, str(output_train),
                   args.tokenizer, args.min_tokens, args.max_tokens)

    print("\n" + "=" * 70 + "\n")

    # Filter dev (usually already short, but filter anyway for consistency)
    output_dev = DATA_DIR / f"qa_dev_filtered_{suffix}.json"
    filter_qa_data(args.input_dev, str(output_dev),
                   args.tokenizer, args.min_tokens, args.max_tokens)

    print("\n" + "=" * 70)
    print("Summary:")
    print(f"  Train: {output_train}")
    print(f"  Dev:   {output_dev}")
    if args.min_tokens > 0:
        print(f"\nUse these files for training with {args.min_tokens} <= doc_tokens <= {args.max_tokens}")
    else:
        print(f"\nUse these files for training with doc_tokens <= {args.max_tokens}")


if __name__ == "__main__":
    main()
