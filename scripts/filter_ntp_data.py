"""Filter NTP training data by document length.

Creates a subset of ntp_train.jsonl containing only documents with
tokenized length < 512 tokens.

Usage:
    python scripts/filter_ntp_data.py \
        --input data/ntp_train.jsonl \
        --output data/ntp_train_512.jsonl \
        --max_length 512
"""

import argparse
import json
import logging
from pathlib import Path

from transformers import AutoTokenizer
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Filter NTP data by token length")
    parser.add_argument("--input", type=str, default="data/ntp_train.jsonl",
                        help="Input JSONL file")
    parser.add_argument("--output", type=str, default="data/ntp_train_512.jsonl",
                        help="Output JSONL file")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Maximum document length in tokens")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B",
                        help="Model name for tokenizer")
    args = parser.parse_args()

    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Reading from: {input_path}")
    logger.info(f"Writing to: {output_path}")
    logger.info(f"Max length: {args.max_length} tokens")

    total_lines = 0
    kept_lines = 0

    with open(input_path, "r", encoding="utf-8") as fin:
        with open(output_path, "w", encoding="utf-8") as fout:
            for line in tqdm(fin, desc="Filtering"):
                total_lines += 1
                try:
                    item = json.loads(line.strip())
                    text = item.get("text", "")

                    # Tokenize and check length
                    tokens = tokenizer.encode(text, add_special_tokens=False)
                    if len(tokens) < args.max_length:
                        fout.write(line)
                        kept_lines += 1
                except Exception as e:
                    logger.warning(f"Error processing line {total_lines}: {e}")
                    continue

    logger.info(f"Total lines: {total_lines:,}")
    logger.info(f"Kept lines: {kept_lines:,} ({100*kept_lines/total_lines:.1f}%)")
    logger.info(f"Filtered data saved to: {output_path}")


if __name__ == "__main__":
    main()
