"""Evaluate all trained checkpoints in outputs/ directory.

Automatically discovers all checkpoint directories, loads models,
and evaluates them on the test set. Shows comprehensive metrics
and sample predictions.

Usage:
    # Evaluate all checkpoints on Stage 1 NTP validation data
    python scripts/evaluate_all_checkpoints.py \
        --eval_data data/ntp_train.jsonl \
        --stage 1

    # Evaluate on Stage 2 QA dev data
    python scripts/evaluate_all_checkpoints.py \
        --eval_data data/qa_dev.json \
        --stage 2

    # Evaluate specific checkpoints
    python scripts/evaluate_all_checkpoints.py \
        --eval_data data/ntp_train_512.jsonl \
        --stage 1 \
        --checkpoints outputs/stage1_q16,outputs/stage1_q32

    # Show more sample predictions
    python scripts/evaluate_all_checkpoints.py \
        --eval_data data/qa_dev.json \
        --stage 2 \
        --show_samples 10

    # Save results to CSV
    python scripts/evaluate_all_checkpoints.py \
        --eval_data data/ntp_train_512.jsonl \
        --stage 1 \
        --output results.csv
"""

import argparse
import csv
import json
import logging
import os
from pathlib import Path
from typing import Dict, List

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from deep_compressor.config import DeepCompressorConfig
from deep_compressor.data import NTPDataset, PaddingCollator, QADataset
from deep_compressor.eval import evaluate_ntp, evaluate_qa
from deep_compressor.model import DeepCompressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_all_checkpoints(base_dir: str = "outputs") -> List[Path]:
    """Find all checkpoint directories in outputs/."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    checkpoints = []
    for checkpoint_dir in base_path.iterdir():
        if checkpoint_dir.is_dir():
            # Look for checkpoint-final
            final_checkpoint = checkpoint_dir / "checkpoint-final" / "trainable_weights.pt"
            if final_checkpoint.exists():
                checkpoints.append(checkpoint_dir)

    return sorted(checkpoints)


def extract_q_value(checkpoint_path: Path) -> int:
    """Extract Q value from checkpoint directory name."""
    name = checkpoint_path.name
    if "stage1_q" in name:
        try:
            return int(name.split("stage1_q")[-1])
        except ValueError:
            pass
    return -1


def load_model(checkpoint_path: Path, config_path: Path,
               accelerator: Accelerator) -> DeepCompressor:
    """Load model from checkpoint."""
    logger.info(f"Loading model from {checkpoint_path}")

    # Load config
    config = DeepCompressorConfig.from_yaml(str(config_path))

    # Create model
    model = DeepCompressor(config)

    # Load weights
    weights_path = checkpoint_path / "checkpoint-final" / "trainable_weights.pt"
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=False)

    # Prepare with accelerator
    model = accelerator.prepare(model)

    return model


def evaluate_checkpoint(checkpoint_dir: Path, eval_data_path: str,
                        stage: int, accelerator: Accelerator,
                        tokenizer, max_eval_samples: int = 0,
                        show_samples: int = 3) -> Dict[str, float]:
    """Evaluate a single checkpoint."""
    # Find config file
    config_name = checkpoint_dir.name
    if "stage1_q" in config_name:
        config_path = Path("configs") / f"{config_name}.yaml"
    else:
        config_path = Path("configs") / "default.yaml"

    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        return {}

    # Load model
    model = load_model(checkpoint_dir, config_path, accelerator)
    config = DeepCompressorConfig.from_yaml(str(config_path))

    # Prepare data
    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
    num_workers = 0 if accelerator.device.type == "mps" else 2
    pin_memory = accelerator.device.type == "cuda"

    if stage == 1:
        # NTP evaluation
        dataset = NTPDataset(
            eval_data_path, tokenizer,
            max_doc_tokens=config.qwen.max_doc_tokens,
            segment_len=config.training.ntp_segment_len,
        )

        # Take validation split (last 10%)
        n_total = len(dataset)
        n_val = min(5000, n_total // 10) if n_total > 10 else n_total
        n_train = n_total - n_val
        val_indices = list(range(n_train, n_total))
        val_subset = Subset(dataset, val_indices)

        if max_eval_samples > 0:
            val_subset = Subset(val_subset, list(range(min(max_eval_samples, len(val_subset)))))

        eval_loader = DataLoader(
            val_subset, batch_size=config.training.batch_size,
            shuffle=False, collate_fn=collator,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        eval_loader = accelerator.prepare(eval_loader)

        logger.info(f"Evaluating on {len(val_subset)} NTP samples")
        metrics = evaluate_ntp(model, eval_loader, accelerator,
                               tokenizer=tokenizer, show_sample=True)

    elif stage == 2:
        # QA evaluation
        dataset = QADataset(
            eval_data_path, tokenizer,
            max_doc_tokens=config.qwen.max_doc_tokens,
            max_question_tokens=config.qwen.max_question_tokens,
            max_answer_tokens=config.qwen.max_answer_tokens,
        )

        if max_eval_samples > 0:
            dataset = Subset(dataset, list(range(min(max_eval_samples, len(dataset)))))

        eval_loader = DataLoader(
            dataset, batch_size=config.training.batch_size,
            shuffle=False, collate_fn=collator,
            num_workers=num_workers, pin_memory=pin_memory,
        )
        eval_loader = accelerator.prepare(eval_loader)

        logger.info(f"Evaluating on {len(dataset)} QA samples")
        metrics = evaluate_qa(
            model, eval_loader, tokenizer, accelerator,
            max_new_tokens=config.qwen.max_answer_tokens,
            show_samples=show_samples,
        )

    else:
        raise ValueError(f"Unknown stage: {stage}")

    return metrics


def save_results(results: List[Dict], output_path: str):
    """Save evaluation results to CSV."""
    if not results:
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["checkpoint", "q_value"] + list(results[0]["metrics"].keys())

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            row = {
                "checkpoint": result["checkpoint"],
                "q_value": result["q_value"],
                **result["metrics"],
            }
            writer.writerow(row)

    logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate all trained checkpoints")
    parser.add_argument("--eval_data", type=str, required=True,
                        help="Path to evaluation data (JSONL for Stage 1, JSON for Stage 2)")
    parser.add_argument("--stage", type=int, required=True, choices=[1, 2],
                        help="Evaluation stage (1=NTP, 2=QA)")
    parser.add_argument("--checkpoints", type=str, default=None,
                        help="Comma-separated list of checkpoint directories (default: auto-discover)")
    parser.add_argument("--max_eval_samples", type=int, default=0,
                        help="Limit eval samples (0 = all)")
    parser.add_argument("--show_samples", type=int, default=3,
                        help="Number of sample predictions to show")
    parser.add_argument("--output", type=str, default=None,
                        help="Save results to CSV file")
    args = parser.parse_args()

    # Initialize accelerator
    accelerator = Accelerator()

    # Find checkpoints
    if args.checkpoints:
        checkpoint_dirs = [Path(p.strip()) for p in args.checkpoints.split(",")]
    else:
        checkpoint_dirs = find_all_checkpoints()

    if not checkpoint_dirs:
        logger.error("No checkpoints found")
        return

    logger.info(f"Found {len(checkpoint_dirs)} checkpoints to evaluate")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "models/Qwen3-0.6B",
        trust_remote_code=True,
        fix_mistral_regex=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Evaluate each checkpoint
    results = []

    for checkpoint_dir in checkpoint_dirs:
        logger.info("\n" + "="*80)
        logger.info(f"Evaluating: {checkpoint_dir.name}")
        logger.info("="*80 + "\n")

        try:
            metrics = evaluate_checkpoint(
                checkpoint_dir=checkpoint_dir,
                eval_data_path=args.eval_data,
                stage=args.stage,
                accelerator=accelerator,
                tokenizer=tokenizer,
                max_eval_samples=args.max_eval_samples,
                show_samples=args.show_samples,
            )

            if metrics:
                q_value = extract_q_value(checkpoint_dir)
                results.append({
                    "checkpoint": checkpoint_dir.name,
                    "q_value": q_value,
                    "metrics": metrics,
                })

                # Print summary
                logger.info(f"\n{'='*80}")
                logger.info(f"Results for {checkpoint_dir.name} (Q={q_value})")
                logger.info(f"{'='*80}")
                for key, value in metrics.items():
                    if isinstance(value, float):
                        logger.info(f"  {key:20s}: {value:.4f}")
                logger.info(f"{'='*80}\n")

        except Exception as e:
            logger.error(f"Failed to evaluate {checkpoint_dir.name}: {e}", exc_info=True)

    # Print comparison table
    if results:
        logger.info("\n" + "="*80)
        logger.info("Comparison Table")
        logger.info("="*80)

        # Sort by Q value
        results.sort(key=lambda x: x["q_value"])

        # Print header
        metric_names = list(results[0]["metrics"].keys())
        header = f"{'Q':<6} | " + " | ".join(f"{m:<10}" for m in metric_names)
        logger.info(header)
        logger.info("-" * len(header))

        # Print rows
        for result in results:
            q = result["q_value"]
            values = [result["metrics"].get(m, float('nan')) for m in metric_names]
            row = f"{q:<6} | " + " | ".join(f"{v:>10.4f}" for v in values)
            logger.info(row)

        logger.info("="*80 + "\n")

    # Save to CSV if requested
    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
