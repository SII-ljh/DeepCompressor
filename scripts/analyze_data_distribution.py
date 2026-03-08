"""Analyze document length distribution in NTP training data.

This script helps you understand the document length distribution
before filtering, so you can make informed decisions about the
max_length threshold.

Usage:
    python scripts/analyze_data_distribution.py \
        --data_path data/ntp_train.jsonl

    # Analyze with specific length thresholds
    python scripts/analyze_data_distribution.py \
        --data_path data/ntp_train.jsonl \
        --thresholds 256,512,1024,2048

    # Save histogram plot
    python scripts/analyze_data_distribution.py \
        --data_path data/ntp_train.jsonl \
        --plot results/length_distribution.png
"""

import argparse
import json
import logging
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def analyze_lengths(data_path: str, tokenizer, thresholds: list = None):
    """Analyze document length distribution."""
    if thresholds is None:
        thresholds = [256, 512, 1024, 2048, 4096, 8192]

    lengths = []
    logger.info(f"Analyzing {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Processing"):
            try:
                item = json.loads(line.strip())
                text = item.get("text", "")
                tokens = tokenizer.encode(text, add_special_tokens=False)
                lengths.append(len(tokens))
            except Exception as e:
                logger.warning(f"Error processing line: {e}")
                continue

    # Basic statistics
    total = len(lengths)
    if total == 0:
        logger.error("No valid samples found")
        return

    sorted_lengths = sorted(lengths)
    mean_length = sum(lengths) / total
    median_length = sorted_lengths[total // 2]
    min_length = sorted_lengths[0]
    max_length = sorted_lengths[-1]

    # Percentiles
    p25 = sorted_lengths[total // 4]
    p75 = sorted_lengths[3 * total // 4]
    p90 = sorted_lengths[int(0.9 * total)]
    p95 = sorted_lengths[int(0.95 * total)]
    p99 = sorted_lengths[int(0.99 * total)]

    # Print summary
    print("\n" + "="*80)
    print("Document Length Statistics")
    print("="*80)
    print(f"Total samples:     {total:,}")
    print(f"Mean length:       {mean_length:.1f} tokens")
    print(f"Median length:     {median_length:,} tokens")
    print(f"Min length:        {min_length:,} tokens")
    print(f"Max length:        {max_length:,} tokens")
    print("")
    print("Percentiles:")
    print(f"  25th percentile: {p25:,} tokens")
    print(f"  75th percentile: {p75:,} tokens")
    print(f"  90th percentile: {p90:,} tokens")
    print(f"  95th percentile: {p95:,} tokens")
    print(f"  99th percentile: {p99:,} tokens")
    print("="*80)

    # Analyze thresholds
    print("\nSamples within thresholds:")
    print("-" * 80)
    print(f"{'Threshold':<15} {'Count':<15} {'Percentage':<15} {'Cumulative %':<15}")
    print("-" * 80)

    for threshold in sorted(thresholds):
        count = sum(1 for l in lengths if l < threshold)
        percentage = 100 * count / total
        print(f"< {threshold:<13} {count:<15,} {percentage:<14.2f}% {percentage:<14.2f}%")

    print("="*80)

    # Distribution by bins
    bins = [0, 128, 256, 512, 1024, 2048, 4096, 8192, float('inf')]
    bin_counts = Counter()

    for length in lengths:
        for i in range(len(bins) - 1):
            if bins[i] <= length < bins[i + 1]:
                bin_label = f"{bins[i]}-{bins[i+1]}" if bins[i+1] != float('inf') else f"{bins[i]}+"
                bin_counts[bin_label] += 1
                break

    print("\nLength distribution by bins:")
    print("-" * 80)
    print(f"{'Bin':<15} {'Count':<15} {'Percentage':<15}")
    print("-" * 80)

    for i in range(len(bins) - 1):
        bin_label = f"{bins[i]}-{bins[i+1]}" if bins[i+1] != float('inf') else f"{bins[i]}+"
        count = bin_counts[bin_label]
        percentage = 100 * count / total
        print(f"{bin_label:<15} {count:<15,} {percentage:<14.2f}%")

    print("="*80 + "\n")

    return lengths


def plot_distribution(lengths: list, output_path: str):
    """Plot length distribution histogram."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        logger.warning("matplotlib not installed, skipping plot")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Full distribution
    axes[0].hist(lengths, bins=100, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Document Length (tokens)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Document Length Distribution (Full Range)')
    axes[0].grid(True, alpha=0.3)

    # Zoomed in (0-2048)
    lengths_filtered = [l for l in lengths if l <= 2048]
    axes[1].hist(lengths_filtered, bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Document Length (tokens)')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Document Length Distribution (0-2048 tokens)')
    axes[1].grid(True, alpha=0.3)

    # Add vertical lines for common thresholds
    for threshold in [256, 512, 1024]:
        axes[1].axvline(threshold, color='red', linestyle='--', linewidth=1, alpha=0.5)
        axes[1].text(threshold, axes[1].get_ylim()[1] * 0.9, f'{threshold}',
                    ha='center', fontsize=8, color='red')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    logger.info(f"Plot saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze document length distribution")
    parser.add_argument("--data_path", type=str, default="data/ntp_train.jsonl",
                        help="Path to NTP training data")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B",
                        help="Model name for tokenizer")
    parser.add_argument("--thresholds", type=str, default="256,512,1024,2048,4096,8192",
                        help="Comma-separated length thresholds to analyze")
    parser.add_argument("--plot", type=str, default=None,
                        help="Save histogram plot to this path")
    args = parser.parse_args()

    # Parse thresholds
    thresholds = [int(t.strip()) for t in args.thresholds.split(",")]

    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Analyze
    lengths = analyze_lengths(args.data_path, tokenizer, thresholds)

    # Plot if requested
    if args.plot and lengths:
        plot_distribution(lengths, args.plot)


if __name__ == "__main__":
    main()
