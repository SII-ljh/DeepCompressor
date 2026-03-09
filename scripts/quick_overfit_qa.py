#!/usr/bin/env python3
"""Quick overfitting test for QA training.

Tests model's capacity to memorize a small QA dataset. If loss doesn't converge
to near zero (~0.01), there's a bug in the training loop or model architecture.

Usage:
    python scripts/quick_overfit_qa.py                    # 10 samples, 1000 steps
    python scripts/quick_overfit_qa.py --samples 5        # 5 samples
    python scripts/quick_overfit_qa.py --steps 2000       # More training steps
    python scripts/quick_overfit_qa.py --q_length 64      # Test different Q
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path

# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from deep_compressor.config import DeepCompressorConfig
from deep_compressor.data import PaddingCollator, QADataset
from deep_compressor.model import DeepCompressor

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


def quick_overfit_test(
    config_path: str,
    data_path: str,
    num_samples: int = 10,
    max_steps: int = 1000,
    batch_size: int = 2,
    q_length: int = None,
):
    """Run quick overfitting test on small QA dataset."""

    # Load config
    config = DeepCompressorConfig.from_yaml(config_path)

    # Override settings for overfitting test
    config.training.batch_size = batch_size
    config.training.max_steps = max_steps
    config.training.gradient_accumulation_steps = 1
    config.training.log_every = 50
    config.training.eval_every = 500
    config.training.save_every = 999999  # No saving
    config.training.warmup_steps = 50
    config.training.learning_rate = 1e-3  # Higher LR for faster convergence

    # Override Q length if specified
    if q_length is not None:
        config.ablation.override_num_queries = q_length

    # Setup accelerator
    accelerator = Accelerator(gradient_accumulation_steps=1)

    logger.info("=" * 70)
    logger.info("QUICK OVERFITTING TEST")
    logger.info("=" * 70)
    logger.info(f"Config: {config_path}")
    logger.info(f"Data: {data_path}")
    logger.info(f"Samples: {num_samples}")
    logger.info(f"Steps: {max_steps}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Q length: {config.effective_num_queries}")
    logger.info(f"Learning rate: {config.training.learning_rate}")
    logger.info("=" * 70)
    logger.info("")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.qwen.model_name_or_path,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    logger.info("Loading model...")
    model = DeepCompressor(config)

    # Enable gradient checkpointing to save memory
    if config.training.gradient_checkpointing:
        model.qwen.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Dataset (limit to num_samples)
    dataset = QADataset(
        data_path,
        tokenizer,
        max_doc_tokens=config.qwen.max_doc_tokens,
        max_question_tokens=config.qwen.max_question_tokens,
        max_answer_tokens=config.qwen.max_answer_tokens,
    )

    # Select random samples
    random.seed(42)
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))
    dataset = Subset(dataset, indices)

    logger.info(f"Dataset: {len(dataset)} samples")

    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=0,
    )

    # Optimizer
    optimizer = AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.training.learning_rate,
        weight_decay=0.0,  # No weight decay for overfitting test
    )

    # Prepare
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)

    # Training loop
    model.train()
    step = 0
    min_loss = float("inf")
    converged = False

    logger.info("Starting training...")
    logger.info("Target: loss should converge to < 0.1 (ideally < 0.01)")
    logger.info("")

    while step < max_steps:
        for batch in loader:
            with accelerator.accumulate(model):
                # Forward
                losses = model(
                    mode="qa",
                    doc_input_ids=batch["doc_input_ids"],
                    doc_attention_mask=batch["doc_attention_mask"],
                    q_input_ids=batch["q_input_ids"],
                    q_attention_mask=batch["q_attention_mask"],
                    answer_ids=batch["answer_ids"],
                    answer_attention_mask=batch["answer_attention_mask"],
                    answer_labels=batch["answer_labels"],
                    global_step=step,
                )

                # Backward
                accelerator.backward(losses["total"])

                # Gradient clipping
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            step += 1

            # Logging
            if step % 50 == 0 or step == 1:
                loss_val = losses["total"].detach().item()
                min_loss = min(min_loss, loss_val)

                # Check convergence
                if loss_val < 0.1:
                    status = "✓ CONVERGED"
                    if not converged:
                        logger.info(f"  → First convergence at step {step}")
                        converged = True
                elif loss_val < 0.5:
                    status = "⚡ Converging"
                elif loss_val < 2.0:
                    status = "📉 Decreasing"
                else:
                    status = "⏳ Training"

                logger.info(
                    f"Step {step:4d}/{max_steps}  |  "
                    f"Loss: {loss_val:.4f}  |  "
                    f"Min: {min_loss:.4f}  |  "
                    f"{status}"
                )

            if step >= max_steps:
                break

    # Final summary
    logger.info("")
    logger.info("=" * 70)
    logger.info("OVERFITTING TEST COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Final loss: {loss_val:.4f}")
    logger.info(f"Min loss:   {min_loss:.4f}")

    if min_loss < 0.01:
        logger.info("✅ PASS: Model successfully memorized the dataset (loss < 0.01)")
    elif min_loss < 0.1:
        logger.info("✅ PASS: Model converged reasonably well (loss < 0.1)")
    elif min_loss < 0.5:
        logger.info("⚠️  PARTIAL: Model converging but not fully (loss < 0.5)")
        logger.info("   → Try: More steps, higher LR, or check model capacity")
    else:
        logger.info("❌ FAIL: Model did not converge (loss >= 0.5)")
        logger.info("   → Check: Model architecture, data loading, or training loop")

    logger.info("=" * 70)


def create_synthetic_qa_data(num_samples: int = 10):
    """Create synthetic QA data for testing without downloading datasets."""
    import tempfile

    # Generate synthetic QA samples
    samples = []
    for i in range(num_samples):
        context = f"This is a sample financial document number {i}. " \
                  f"The company reported revenue of ${100 + i*10} million in Q{i%4 + 1}. " \
                  f"The CEO is John Smith and the CFO is Jane Doe. " \
                  f"The company operates in the technology sector with {50 + i*5} employees. " \
                  f"Their main product is Software Solution {i} which generated ${20 + i*2} million."

        question = f"What was the revenue in Q{i%4 + 1}?"
        answer = f"${100 + i*10} million"

        samples.append({
            "context": context,
            "question": question,
            "answer": answer,
        })

    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False)
    json.dump(samples, temp_file, ensure_ascii=False)
    temp_file.close()

    logger.info(f"✓ Created synthetic data: {temp_file.name}")
    logger.info(f"  {len(samples)} samples generated")

    return temp_file.name


def main():
    parser = argparse.ArgumentParser(description="Quick QA overfitting test")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stage1_guided_q128.yaml",
        help="Config file",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="QA data path (default: data/qa_tiny_train.json if exists, else data/qa_train.json)",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic data (no download required)",
    )
    parser.add_argument(
        "--samples", type=int, default=10, help="Number of samples to overfit"
    )
    parser.add_argument(
        "--steps", type=int, default=1000, help="Maximum training steps"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size"
    )
    parser.add_argument(
        "--q_length", type=int, default=None, help="Override Q length (num_queries)"
    )

    args = parser.parse_args()

    # Use synthetic data if requested
    if args.synthetic:
        logger.info("Using synthetic data (no download required)")
        args.data = create_synthetic_qa_data(num_samples=args.samples)
    # Auto-detect data path
    elif args.data is None:
        tiny_path = DATA_DIR / "qa_tiny_train.json"
        full_path = DATA_DIR / "qa_train.json"
        large_path = DATA_DIR / "qa_large_train.json"

        if tiny_path.exists():
            args.data = str(tiny_path)
        elif large_path.exists():
            args.data = str(large_path)
        elif full_path.exists():
            args.data = str(full_path)
        else:
            logger.error("=" * 70)
            logger.error("ERROR: No QA data found!")
            logger.error("=" * 70)
            logger.error("")
            logger.error("Please choose one of the following options:")
            logger.error("")
            logger.error("Option 1 (FASTEST): Use synthetic data (no download)")
            logger.error("  python scripts/quick_overfit_qa.py --synthetic")
            logger.error("")
            logger.error("Option 2: Download small test dataset (~10K samples, ~5 min)")
            logger.error("  python scripts/prepare_large_qa_data.py --test")
            logger.error("")
            logger.error("Option 3: Download large dataset (~800K samples, ~1-2 hours)")
            logger.error("  python scripts/prepare_large_qa_data.py")
            logger.error("")
            return

    quick_overfit_test(
        config_path=args.config,
        data_path=args.data,
        num_samples=args.samples,
        max_steps=args.steps,
        batch_size=args.batch_size,
        q_length=args.q_length,
    )


if __name__ == "__main__":
    main()
