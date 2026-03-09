#!/usr/bin/env python3
"""Filter QA dataset by document length.

Keeps only samples where context length <= max_length.

Usage:
    python scripts/filter_qa_by_length.py \
        --input data/qa_large_train.json \
        --output data/qa_large_train_512.json \
        --max_length 512
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from transformers import AutoTokenizer
    HAS_TOKENIZER = True
except ImportError:
    HAS_TOKENIZER = False
    print("Warning: transformers not installed, using character count")


def filter_dataset(input_path: str, output_path: str, max_length: int,
                   tokenizer=None):
    """Filter dataset by context length."""
    print(f"Loading dataset from {input_path}...")

    with open(input_path, 'r') as f:
        data = json.load(f)

    print(f"✓ Loaded {len(data):,} samples")
    print(f"  Filtering: context <= {max_length} {('tokens' if tokenizer else 'characters')}")
    print()

    # Filter samples
    filtered = []
    length_distribution = []

    for i, sample in enumerate(data):
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1:,}/{len(data):,}...")

        context = sample.get('context', '')

        # Calculate length
        if tokenizer:
            ctx_len = len(tokenizer.encode(context, add_special_tokens=False))
        else:
            ctx_len = len(context)

        length_distribution.append(ctx_len)

        if ctx_len <= max_length:
            filtered.append(sample)

    print(f"✓ Filtered {len(filtered):,}/{len(data):,} samples "
          f"({100*len(filtered)/len(data):.1f}% kept)")
    print()

    # Statistics
    import numpy as np
    lengths = np.array(length_distribution)

    print("Length statistics (before filtering):")
    print(f"  Min:    {lengths.min():,}")
    print(f"  Median: {int(np.median(lengths)):,}")
    print(f"  Mean:   {int(lengths.mean()):,}")
    print(f"  95th:   {int(np.percentile(lengths, 95)):,}")
    print(f"  Max:    {lengths.max():,}")
    print()

    print(f"Samples > {max_length}: {(lengths > max_length).sum():,} "
          f"({100*(lengths > max_length).sum()/len(lengths):.1f}% discarded)")
    print()

    # Save filtered data
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(filtered, f, ensure_ascii=False)

    print(f"✓ Saved {len(filtered):,} samples to {output_path}")
    print(f"  File size: {output_path.stat().st_size / 1e6:.1f} MB")


def main():
    parser = argparse.ArgumentParser(
        description="Filter QA dataset by context length"
    )
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--tokenizer", type=str, default="models/Qwen3-0.6B")

    args = parser.parse_args()

    # Load tokenizer
    tokenizer = None
    if HAS_TOKENIZER:
        try:
            print(f"Loading tokenizer from {args.tokenizer}...")
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer, trust_remote_code=True)
            print("✓ Tokenizer loaded\n")
        except Exception as e:
            print(f"Warning: {e}\nUsing character count\n")

    filter_dataset(args.input, args.output, args.max_length, tokenizer)


if __name__ == "__main__":
    main()
