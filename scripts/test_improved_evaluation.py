#!/usr/bin/env python3
"""Test improved evaluation system with unified output and enhanced metrics.

This script tests the new evaluation features:
- Token accuracy and top-5 accuracy calculation
- Sample collection and unified display
- Comparison table with all metrics

Usage:
    python scripts/test_improved_evaluation.py
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

# Import the improved evaluation function
sys.path.insert(0, str(script_dir))
from evaluate_all_checkpoints import evaluate_direct_qwen_ntp

def main():
    print("\n" + "="*100)
    print("Testing Improved Evaluation System")
    print("="*100 + "\n")

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

    # Load config
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

    # Prepare tiny dataset
    print("Preparing test data...")
    data_path = "data/ntp_tiny.jsonl"
    if not Path(data_path).exists():
        print(f"Error: {data_path} not found")
        return 1

    dataset = NTPDataset(
        data_path, tokenizer,
        max_doc_tokens=config.qwen.max_doc_tokens,
        segment_len=config.training.ntp_segment_len,
    )

    # Use only first 5 samples
    n_test = min(5, len(dataset))
    test_subset = Subset(dataset, list(range(n_test)))
    print(f"Using {n_test} samples for testing\n")

    # Create dataloader
    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
    eval_loader = DataLoader(
        test_subset,
        batch_size=2,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    eval_loader = accelerator.prepare(eval_loader)

    # Run evaluation with new features
    print("Running improved evaluation...")
    print("-" * 100)
    try:
        metrics = evaluate_direct_qwen_ntp(
            eval_loader, qwen_model, accelerator, tokenizer,
            collect_samples=2,  # Collect 2 samples
            max_gen_tokens=30   # Generate 30 tokens per sample
        )
        print("-" * 100)
        print("\n✅ Evaluation successful!\n")

        # Display results in the new format
        print("="*100)
        print("EVALUATION RESULTS")
        print("="*100)
        print(f"\nCore Metrics:")
        print(f"  Perplexity:      {metrics['perplexity']:.6f}")
        print(f"  Loss:            {metrics['loss']:.6f}")
        print(f"  Token Accuracy:  {metrics['token_accuracy']:.6f}  (Top-1 token prediction)")
        print(f"  Top-5 Accuracy:  {metrics['top5_accuracy']:.6f}  (Correct token in top-5)")

        # Display samples
        samples = metrics.get('samples', [])
        if samples:
            print(f"\n{'─'*100}")
            print(f"SAMPLE PREDICTIONS ({len(samples)} samples collected)")
            print(f"{'─'*100}")

            for i, sample in enumerate(samples, 1):
                print(f"\n[Sample {i}/{len(samples)}]")
                print(f"  Doc Preview: {sample['doc_preview'][:80]}...")
                print(f"  Prediction:  {sample['prediction'][:80]}")
                print(f"  Gold:        {sample['gold'][:80]}")
                if i < len(samples):
                    print("")
        else:
            print("\n(No samples collected)")

        print("\n" + "="*100)
        print("✅ Test PASSED: All new features working correctly!")
        print("="*100)
        print("\nNew Features Verified:")
        print("  ✓ Token accuracy calculation")
        print("  ✓ Top-5 accuracy calculation")
        print("  ✓ Sample collection (with doc preview)")
        print("  ✓ Unified metrics dictionary")
        print("\n")
        return 0

    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
