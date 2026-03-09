#!/usr/bin/env python3
"""Pure evaluation script (no training) for quick verification.

Usage:
    python scripts/eval_only.py \\
        --config configs/qa_q256_8gpu.yaml \\
        --checkpoint outputs/qa_q256_8gpu/checkpoint-final \\
        --eval_data data/qa_large_dev.json \\
        --max_samples 512
"""
import argparse
import logging
import os

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from deep_compressor.config import DeepCompressorConfig
from deep_compressor.data import PaddingCollator, QADataset
from deep_compressor.eval import evaluate_qa
from deep_compressor.model import DeepCompressor

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(name)s - %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Pure evaluation (no training)")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--eval_data", type=str, required=True,
                        help="Path to QA eval data (JSON)")
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Limit eval samples (0 = all)")
    parser.add_argument("--show_samples", type=int, default=5,
                        help="Number of sample predictions to display")
    args = parser.parse_args()

    # Load config
    config = DeepCompressorConfig.from_yaml(args.config)

    # Setup accelerator
    accelerator = Accelerator(mixed_precision=config.training.mixed_precision)

    if accelerator.is_main_process:
        logger.info("="*70)
        logger.info("Pure Evaluation Mode (No Training)")
        logger.info("="*70)
        logger.info(f"Config: {args.config}")
        logger.info(f"Checkpoint: {args.checkpoint}")
        logger.info(f"Eval data: {args.eval_data}")
        logger.info(f"Max samples: {args.max_samples if args.max_samples > 0 else 'all'}")
        logger.info(f"Device: {accelerator.device}")
        logger.info(f"Processes: {accelerator.num_processes}")
        logger.info("="*70)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.qwen.model_name_or_path,
        trust_remote_code=True,
        fix_mistral_regex=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    model = DeepCompressor(config)

    # Load checkpoint weights
    weights = torch.load(
        os.path.join(args.checkpoint, "trainable_weights.pt"),
        map_location="cpu",
        weights_only=True
    )
    model.load_state_dict(weights, strict=False)
    logger.info(f"Loaded checkpoint from {args.checkpoint}")

    # Prepare model with accelerator
    model = accelerator.prepare(model)

    # Load eval dataset
    eval_ds = QADataset(
        args.eval_data, tokenizer,
        max_doc_tokens=config.qwen.max_doc_tokens,
        max_question_tokens=config.qwen.max_question_tokens,
        max_answer_tokens=config.qwen.max_answer_tokens
    )

    if args.max_samples > 0 and args.max_samples < len(eval_ds):
        eval_ds = Subset(eval_ds, list(range(args.max_samples)))

    if accelerator.is_main_process:
        logger.info(f"Eval dataset: {len(eval_ds):,} samples")

    # Create dataloader
    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
    num_workers = 0 if accelerator.device.type == "mps" else 2
    pin_memory = accelerator.device.type == "cuda"

    eval_loader = DataLoader(
        eval_ds,
        batch_size=config.training.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    eval_loader = accelerator.prepare(eval_loader)

    # Run evaluation
    if accelerator.is_main_process:
        logger.info("Starting evaluation...")

    metrics = evaluate_qa(
        model, eval_loader, tokenizer, accelerator,
        max_new_tokens=config.qwen.max_answer_tokens,
        show_samples=args.show_samples
    )

    # Print results
    if accelerator.is_main_process:
        logger.info("="*70)
        logger.info("EVALUATION RESULTS")
        logger.info("="*70)
        logger.info(f"Loss:         {metrics['loss']:.4f}")
        logger.info(f"Perplexity:   {metrics['perplexity']:.2f}")
        logger.info(f"Exact Match:  {metrics['exact_match']:.2%}")
        logger.info(f"F1 Score:     {metrics['f1']:.4f}")
        logger.info("="*70)

        # Interpretation
        logger.info("\nInterpretation:")
        if metrics['loss'] < 5.0:
            logger.info("✓ Loss looks normal (< 5.0)")
        elif metrics['loss'] < 10.0:
            logger.info("⚠ Loss is high but reasonable (5-10)")
        else:
            logger.info("✗ Loss is abnormally high (> 10) - possible bug!")

        if metrics['perplexity'] < 100:
            logger.info("✓ Perplexity looks normal (< 100)")
        elif metrics['perplexity'] < 1000:
            logger.info("⚠ Perplexity is high but reasonable (100-1000)")
        else:
            logger.info("✗ Perplexity is abnormally high (> 1000) - possible bug!")

        logger.info("\nExpected ranges (for Q=256, single-stage training):")
        logger.info("  Loss:       2.5 - 4.0 (normal)")
        logger.info("  Perplexity: 12 - 55 (normal)")
        logger.info("  EM:         2% - 5% (low due to high compression)")
        logger.info("  F1:         0.08 - 0.15 (low due to high compression)")


if __name__ == "__main__":
    main()
