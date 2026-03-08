"""Batch training script for Stage 1 with varying query counts.

Trains 5 models with different num_queries: 16, 32, 64, 128, 256.
All models use the same filtered training data (doc_length < 512 tokens).

Usage:
    # Train all 5 models sequentially
    python scripts/train_stage1_varying_q.py

    # Train specific Q values
    python scripts/train_stage1_varying_q.py --q_values 16,32,64

    # Resume training
    python scripts/train_stage1_varying_q.py --resume

    # Dry run (print commands only)
    python scripts/train_stage1_varying_q.py --dry_run
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def check_data_exists(data_path: str) -> bool:
    """Check if the filtered training data exists."""
    return Path(data_path).exists()


def check_checkpoint_exists(output_dir: str) -> bool:
    """Check if a checkpoint already exists for this Q value."""
    checkpoint_path = Path(output_dir) / "checkpoint-final" / "trainable_weights.pt"
    return checkpoint_path.exists()


def run_training(q_value: int, data_path: str, resume: bool = False,
                 dry_run: bool = False, use_wandb: bool = True):
    """Run training for a single Q value."""
    config_file = f"configs/stage1_q{q_value}.yaml"
    output_dir = f"outputs/stage1_q{q_value}"

    # Check if checkpoint exists
    if not resume and check_checkpoint_exists(output_dir):
        logger.warning(f"Checkpoint already exists for Q={q_value} at {output_dir}")
        logger.warning("Skipping training. Use --resume to continue training.")
        return False

    # Build command
    cmd = [
        sys.executable, "-m", "deep_compressor.train",
        "--config", config_file,
        "--data_path", data_path,
        "--stage", "1",
    ]

    if resume:
        resume_path = f"{output_dir}/checkpoint-final"
        if Path(resume_path).exists():
            cmd.extend(["--resume_from", resume_path])
            logger.info(f"Resuming from checkpoint: {resume_path}")

    if use_wandb:
        cmd.append("--wandb")
        cmd.extend(["--wandb_project", "deep-compressor"])

    logger.info(f"{'[DRY RUN] ' if dry_run else ''}Training Q={q_value}")
    logger.info(f"Command: {' '.join(cmd)}")

    if dry_run:
        return True

    # Run training
    try:
        start_time = datetime.now()
        result = subprocess.run(cmd, check=True)
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds() / 3600
        logger.info(f"Training Q={q_value} completed successfully in {elapsed:.2f} hours")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Training Q={q_value} failed with error code {e.returncode}")
        return False
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        raise


def main():
    parser = argparse.ArgumentParser(
        description="Batch training for Stage 1 with varying query counts")
    parser.add_argument("--q_values", type=str, default="16,32,64,128,256",
                        help="Comma-separated Q values to train (default: 16,32,64,128,256)")
    parser.add_argument("--data_path", type=str, default="data/ntp_train.jsonl",
                        help="Path to training data (will auto-truncate to max_doc_tokens)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from existing checkpoints")
    parser.add_argument("--dry_run", action="store_true",
                        help="Print commands without executing")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Disable wandb logging")
    args = parser.parse_args()

    # Parse Q values
    q_values = [int(q.strip()) for q in args.q_values.split(",")]
    logger.info(f"Training models with Q values: {q_values}")

    # Check if data exists
    if not check_data_exists(args.data_path):
        logger.error(f"Training data not found: {args.data_path}")
        logger.error("Please run: python scripts/filter_ntp_data.py first")
        sys.exit(1)

    logger.info(f"Using training data: {args.data_path}")

    # Train each model
    successful = []
    failed = []

    for q_value in q_values:
        logger.info("\n" + "="*80)
        logger.info(f"Starting training for Q={q_value}")
        logger.info("="*80 + "\n")

        try:
            success = run_training(
                q_value=q_value,
                data_path=args.data_path,
                resume=args.resume,
                dry_run=args.dry_run,
                use_wandb=not args.no_wandb,
            )
            if success:
                successful.append(q_value)
            else:
                failed.append(q_value)
        except KeyboardInterrupt:
            logger.warning("\n\nTraining interrupted by user")
            break

    # Summary
    logger.info("\n" + "="*80)
    logger.info("Training Summary")
    logger.info("="*80)
    logger.info(f"Successful: {successful}")
    logger.info(f"Failed: {failed}")
    logger.info("="*80)


if __name__ == "__main__":
    main()
