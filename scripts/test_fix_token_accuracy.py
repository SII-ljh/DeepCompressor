#!/usr/bin/env python3
"""Test the token accuracy fix for dimension mismatch bug.

This script verifies that the token accuracy calculation correctly handles
the prefix + segment logits from the decoder.

Usage:
    python scripts/test_fix_token_accuracy.py
"""

import sys
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from deep_compressor.config import DeepCompressorConfig
from deep_compressor.data import NTPDataset, PaddingCollator
from deep_compressor.model import DeepCompressor
from deep_compressor.eval import evaluate_ntp

def main():
    print("\n" + "="*80)
    print("Testing Token Accuracy Fix (Dimension Mismatch Bug)")
    print("="*80 + "\n")

    # Check if there's a trained model
    checkpoint_dirs = [
        Path("outputs/stage1_q16"),
        Path("outputs/stage1_q32"),
        Path("outputs/h200_stage1"),
    ]

    checkpoint_dir = None
    for ckpt in checkpoint_dirs:
        if (ckpt / "checkpoint-final" / "trainable_weights.pt").exists():
            checkpoint_dir = ckpt
            break

    if checkpoint_dir is None:
        print("❌ No trained checkpoint found. Please run training first.")
        print("   Looking for: outputs/stage1_q*/checkpoint-final/trainable_weights.pt")
        return 1

    print(f"Using checkpoint: {checkpoint_dir}\n")

    # Initialize
    accelerator = Accelerator()
    device = accelerator.device
    print(f"Device: {device}")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "models/Qwen3-0.6B",
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load config
    print("Loading config...")
    config_name = checkpoint_dir.name
    if "stage1_q" in config_name:
        config_path = Path("configs") / f"{config_name}.yaml"
    else:
        config_path = Path("configs") / "default.yaml"

    if not config_path.exists():
        print(f"❌ Config not found: {config_path}")
        return 1

    config = DeepCompressorConfig.from_yaml(str(config_path))
    print(f"Config: {config_path}")
    print(f"  - num_queries: {config.effective_num_queries}")
    print(f"  - max_doc_tokens: {config.qwen.max_doc_tokens}")

    # Load model
    print("\nLoading model...")
    model = DeepCompressor(config)
    weights_path = checkpoint_dir / "checkpoint-final" / "trainable_weights.pt"
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=False)
    model = accelerator.prepare(model)
    print("✓ Model loaded")

    # Prepare tiny dataset (only 2 samples for quick test)
    print("\nPreparing test data...")
    data_path = "data/ntp_tiny.jsonl"
    if not Path(data_path).exists():
        data_path = "data/ntp_train.jsonl"

    dataset = NTPDataset(
        data_path, tokenizer,
        max_doc_tokens=config.qwen.max_doc_tokens,
        segment_len=config.training.ntp_segment_len,
    )

    # Use only 2 samples
    n_test = min(2, len(dataset))
    test_subset = Subset(dataset, list(range(n_test)))
    print(f"Using {n_test} samples for testing")

    # Create dataloader
    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
    eval_loader = DataLoader(
        test_subset,
        batch_size=1,  # Batch size 1 to simplify debugging
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    eval_loader = accelerator.prepare(eval_loader)

    # Run evaluation (this should NOT crash with dimension mismatch)
    print("\nRunning evaluation with token accuracy calculation...")
    print("-" * 80)

    try:
        metrics = evaluate_ntp(
            model, eval_loader, accelerator,
            tokenizer=tokenizer,
            collect_samples=1,  # Collect 1 sample
            max_gen_tokens=20   # Generate 20 tokens
        )
        print("-" * 80)
        print("\n✅ Evaluation successful! Bug is FIXED!")

        print("\nMetrics:")
        print(f"  Perplexity:      {metrics['perplexity']:.6f}")
        print(f"  Loss:            {metrics['loss']:.6f}")
        print(f"  Token Accuracy:  {metrics['token_accuracy']:.6f}")
        print(f"  Top-5 Accuracy:  {metrics['top5_accuracy']:.6f}")

        if metrics.get('samples'):
            print(f"\n✓ Sample collection also working ({len(metrics['samples'])} samples)")

        print("\n" + "="*80)
        print("✅ Test PASSED: Token accuracy calculation fixed!")
        print("="*80)
        print("\nThe dimension mismatch bug has been resolved.")
        print("You can now run: bash scripts/evaluate_all_models.sh")
        print("\n")
        return 0

    except RuntimeError as e:
        if "size of tensor" in str(e):
            print(f"\n❌ Test FAILED: Dimension mismatch still exists!")
            print(f"   Error: {e}")
            print("\nThis means the fix didn't work. Please check the code.")
            return 1
        else:
            raise

if __name__ == "__main__":
    sys.exit(main())
