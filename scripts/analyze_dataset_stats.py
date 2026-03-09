#!/usr/bin/env python3
"""Analyze QA dataset statistics: size, length distribution, etc.

Usage:
    python scripts/analyze_dataset_stats.py
    python scripts/analyze_dataset_stats.py --data data/qa_large_train.json
    python scripts/analyze_dataset_stats.py --output results/dataset_stats.json
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter
from typing import Dict, List

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from transformers import AutoTokenizer
    HAS_TOKENIZER = True
except ImportError:
    HAS_TOKENIZER = False
    print("Warning: transformers not installed, will use character count instead of token count")


def analyze_dataset(data_path: str, tokenizer=None):
    """Analyze dataset and return statistics."""
    print(f"Loading dataset from {data_path}...")

    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f"✓ Loaded {len(data):,} samples")
    print()

    # Statistics to collect
    stats = {
        'total_samples': len(data),
        'context_lengths': [],
        'question_lengths': [],
        'answer_lengths': [],
        'sources': [],
    }

    # Analyze each sample
    print("Analyzing samples...")
    for i, sample in enumerate(data):
        if (i + 1) % 10000 == 0:
            print(f"  Processed {i+1:,}/{len(data):,} samples...")

        context = sample.get('context', '')
        question = sample.get('question', '')
        answer = sample.get('answer', '')
        source = sample.get('source', 'unknown')

        # Calculate lengths
        if tokenizer:
            ctx_len = len(tokenizer.encode(context, add_special_tokens=False))
            q_len = len(tokenizer.encode(question, add_special_tokens=False))
            a_len = len(tokenizer.encode(answer, add_special_tokens=False))
        else:
            ctx_len = len(context)
            q_len = len(question)
            a_len = len(answer)

        stats['context_lengths'].append(ctx_len)
        stats['question_lengths'].append(q_len)
        stats['answer_lengths'].append(a_len)
        stats['sources'].append(source)

    print(f"✓ Analyzed all {len(data):,} samples")
    print()

    # Compute statistics
    def compute_stats(lengths, name):
        arr = np.array(lengths)
        return {
            f'{name}_min': int(arr.min()),
            f'{name}_max': int(arr.max()),
            f'{name}_mean': float(arr.mean()),
            f'{name}_median': float(np.median(arr)),
            f'{name}_p25': float(np.percentile(arr, 25)),
            f'{name}_p75': float(np.percentile(arr, 75)),
            f'{name}_p95': float(np.percentile(arr, 95)),
            f'{name}_p99': float(np.percentile(arr, 99)),
        }

    result = {
        'dataset': data_path,
        'total_samples': stats['total_samples'],
        'unit': 'tokens' if tokenizer else 'characters',
    }

    result.update(compute_stats(stats['context_lengths'], 'context'))
    result.update(compute_stats(stats['question_lengths'], 'question'))
    result.update(compute_stats(stats['answer_lengths'], 'answer'))

    # Source distribution
    source_counts = Counter(stats['sources'])
    result['source_distribution'] = dict(source_counts)

    return result, stats


def print_stats(result: Dict, stats: Dict):
    """Print statistics in a readable format."""
    print("=" * 80)
    print("DATASET STATISTICS")
    print("=" * 80)
    print()

    print(f"Dataset: {result['dataset']}")
    print(f"Total samples: {result['total_samples']:,}")
    print(f"Unit: {result['unit']}")
    print()

    # Context length
    print("-" * 80)
    print("CONTEXT LENGTH DISTRIBUTION")
    print("-" * 80)
    print(f"  Min:       {result['context_min']:8,} {result['unit']}")
    print(f"  25th %ile: {result['context_p25']:8,.0f} {result['unit']}")
    print(f"  Median:    {result['context_median']:8,.0f} {result['unit']}")
    print(f"  Mean:      {result['context_mean']:8,.0f} {result['unit']}")
    print(f"  75th %ile: {result['context_p75']:8,.0f} {result['unit']}")
    print(f"  95th %ile: {result['context_p95']:8,.0f} {result['unit']}")
    print(f"  99th %ile: {result['context_p99']:8,.0f} {result['unit']}")
    print(f"  Max:       {result['context_max']:8,} {result['unit']}")
    print()

    # Question length
    print("-" * 80)
    print("QUESTION LENGTH DISTRIBUTION")
    print("-" * 80)
    print(f"  Min:       {result['question_min']:8,} {result['unit']}")
    print(f"  Median:    {result['question_median']:8,.0f} {result['unit']}")
    print(f"  Mean:      {result['question_mean']:8,.0f} {result['unit']}")
    print(f"  95th %ile: {result['question_p95']:8,.0f} {result['unit']}")
    print(f"  Max:       {result['question_max']:8,} {result['unit']}")
    print()

    # Answer length
    print("-" * 80)
    print("ANSWER LENGTH DISTRIBUTION")
    print("-" * 80)
    print(f"  Min:       {result['answer_min']:8,} {result['unit']}")
    print(f"  Median:    {result['answer_median']:8,.0f} {result['unit']}")
    print(f"  Mean:      {result['answer_mean']:8,.0f} {result['unit']}")
    print(f"  95th %ile: {result['answer_p95']:8,.0f} {result['unit']}")
    print(f"  Max:       {result['answer_max']:8,} {result['unit']}")
    print()

    # Source distribution
    print("-" * 80)
    print("SOURCE DISTRIBUTION")
    print("-" * 80)
    total = result['total_samples']
    for source, count in sorted(result['source_distribution'].items(),
                                 key=lambda x: x[1], reverse=True):
        pct = 100 * count / total
        print(f"  {source:20s}: {count:8,} ({pct:5.1f}%)")
    print()

    print("=" * 80)


def plot_distribution(stats: Dict, output_path: str):
    """Plot length distributions (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        sns.set_style("whitegrid")
    except ImportError:
        print("Warning: matplotlib/seaborn not installed, skipping plots")
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Context length
    axes[0].hist(stats['context_lengths'], bins=50, alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Context Length')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Context Length Distribution')
    axes[0].axvline(np.median(stats['context_lengths']), color='r',
                    linestyle='--', label='Median')
    axes[0].legend()

    # Question length
    axes[1].hist(stats['question_lengths'], bins=50, alpha=0.7,
                 edgecolor='black', color='orange')
    axes[1].set_xlabel('Question Length')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Question Length Distribution')
    axes[1].axvline(np.median(stats['question_lengths']), color='r',
                    linestyle='--', label='Median')
    axes[1].legend()

    # Answer length
    axes[2].hist(stats['answer_lengths'], bins=50, alpha=0.7,
                 edgecolor='black', color='green')
    axes[2].set_xlabel('Answer Length')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title('Answer Length Distribution')
    axes[2].axvline(np.median(stats['answer_lengths']), color='r',
                    linestyle='--', label='Median')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze QA dataset statistics"
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to QA JSON file (default: auto-detect)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/dataset_stats.json",
        help="Output path for stats JSON"
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Output path for distribution plot (PNG)"
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="models/Qwen3-0.6B",
        help="Tokenizer path (default: models/Qwen3-0.6B)"
    )

    args = parser.parse_args()

    # Auto-detect data path
    if args.data is None:
        candidates = [
            "data/qa_large_train.json",
            "data/qa_train.json",
        ]
        for path in candidates:
            if Path(path).exists():
                args.data = path
                break

        if args.data is None:
            print("Error: No QA data found. Please specify --data")
            print("\nTried:")
            for p in candidates:
                print(f"  - {p}")
            return

    # Load tokenizer
    tokenizer = None
    if HAS_TOKENIZER:
        try:
            print(f"Loading tokenizer from {args.tokenizer}...")
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer,
                trust_remote_code=True,
            )
            print("✓ Tokenizer loaded")
            print()
        except Exception as e:
            print(f"Warning: Failed to load tokenizer: {e}")
            print("Using character count instead")
            print()

    # Analyze dataset
    result, stats = analyze_dataset(args.data, tokenizer)

    # Print statistics
    print_stats(result, stats)

    # Save to JSON
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved statistics to {output_path}")
    print()

    # Plot if requested
    if args.plot:
        plot_path = Path(args.plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        plot_distribution(stats, str(plot_path))
        print()


if __name__ == "__main__":
    main()
