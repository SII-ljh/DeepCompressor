#!/usr/bin/env python3
"""Test script to verify direct_qwen baseline evaluation works correctly.

This script tests ONLY the baseline evaluation (without any trained models)
using a small subset of data to quickly verify functionality.

Usage:
    python scripts/test_baseline_only.py
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
from transformers import AutoModelForCausalLM, AutoTokenizer

from deep_compressor.config import DeepCompressorConfig
from deep_compressor.data import NTPDataset, PaddingCollator

# Import the baseline evaluation function
sys.path.insert(0, str(script_dir))
from evaluate_all_checkpoints import evaluate_direct_qwen_ntp

def main():
    print("\n" + "="*80)
    print("Testing Direct Qwen Baseline Evaluation")
    print("="*80 + "\n")

    # Initialize accelerator
    accelerator = Accelerator()
    device = accelerator.device
    print(f"Device: {device}\n")

    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "models/Qwen3-0.6B",
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load config (for data params)
    print("Loading config...")
    config_path = "configs/tiny_subset.yaml"
    if not Path(config_path).exists():
        config_path = "configs/default.yaml"
    config = DeepCompressorConfig.from_yaml(config_path)

    # Load Qwen model
    print("Loading Qwen3-0.6B model...")
    qwen_model = AutoModelForCausalLM.from_pretrained(
        "models/Qwen3-0.6B",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    qwen_model.eval()
    for p in qwen_model.parameters():
        p.requires_grad = False
    qwen_model = accelerator.prepare(qwen_model)
    print("✓ Model loaded\n")

    # Prepare tiny dataset (only 10 samples for quick test)
    print("Preparing test data...")
    data_path = "data/ntp_tiny.jsonl"
    if not Path(data_path).exists():
        print(f"Error: {data_path} not found")
        print("Please run: python scripts/prepare_data.py --make-tiny")
        return 1

    dataset = NTPDataset(
        data_path, tokenizer,
        max_doc_tokens=config.qwen.max_doc_tokens,
        segment_len=config.training.ntp_segment_len,
    )

    # Use only first 10 samples for quick test
    n_test = min(10, len(dataset))
    test_subset = Subset(dataset, list(range(n_test)))
    print(f"Using {n_test} samples for testing\n")

    # Create dataloader
    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
    num_workers = 0 if device.type == "mps" else 0  # Use 0 for testing
    pin_memory = device.type == "cuda"

    eval_loader = DataLoader(
        test_subset,
        batch_size=2,  # Small batch for testing
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    eval_loader = accelerator.prepare(eval_loader)

    # Run baseline evaluation
    print("Running baseline evaluation...")
    print("-" * 80)
    try:
        metrics = evaluate_direct_qwen_ntp(eval_loader, qwen_model, accelerator)
        print("-" * 80)
        print("\n✅ Baseline evaluation successful!\n")
        print("Results:")
        print(f"  Perplexity: {metrics['perplexity']:.4f}")
        print(f"  Loss:       {metrics['loss']:.4f}")
        print("\n" + "="*80)
        print("Test PASSED: Direct Qwen baseline evaluation works correctly")
        print("="*80 + "\n")
        return 0

    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
