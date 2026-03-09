#!/usr/bin/env python3
"""Train QA models with varying Q (num_queries) values.

Trains multiple models directly on QA task (without Stage 1 NTP pretraining)
to find the optimal compression ratio for different document lengths.

Usage:
    # Train all Q values
    python scripts/train_qa_varying_q.py

    # Train specific Q values
    python scripts/train_qa_varying_q.py --q_values 32 64 128

    # Use large QA dataset
    python scripts/train_qa_varying_q.py --use_large_data

    # Dry run (print commands only)
    python scripts/train_qa_varying_q.py --dry_run
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
DATA_DIR = PROJECT_ROOT / "data"


def create_qa_config(q_value: int, base_config: str = "configs/default.yaml",
                     output_dir: str = None):
    """Create a QA training config for specific Q value.

    Reads base config and overrides:
    - perceiver.num_queries = q_value
    - training.output_dir = outputs/qa_q{q_value}
    - wandb.run_name = qa_q{q_value}
    """
    import yaml

    with open(base_config) as f:
        config = yaml.safe_load(f)

    # Override Q value
    config["perceiver"]["num_queries"] = q_value

    # Override output directory
    if output_dir is None:
        output_dir = f"outputs/qa_q{q_value}"
    config["training"]["output_dir"] = output_dir
    config["training"]["stage"] = 2  # QA training

    # Override wandb settings
    if "wandb" not in config:
        config["wandb"] = {}
    config["wandb"]["enabled"] = True
    config["wandb"]["project"] = "deep-compressor-qa"
    config["wandb"]["run_name"] = f"qa_q{q_value}"
    config["wandb"]["tags"] = ["qa_only", f"q{q_value}"]

    # Save config
    config_path = CONFIGS_DIR / f"qa_q{q_value}.yaml"
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"  ✓ Created config: {config_path}")
    return config_path


def train_qa_model(config_path: str, data_path: str, eval_data_path: str,
                   dry_run: bool = False):
    """Train a single QA model."""
    cmd = [
        "python", "-m", "deep_compressor.train",
        "--config", str(config_path),
        "--data_path", str(data_path),
        "--eval_data_path", str(eval_data_path),
        "--stage", "2",
        "--wandb", "--wandb_project", "deep-compressor-qa",
    ]

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Running command:")
    print(" ".join(cmd))
    print()

    if not dry_run:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"❌ Training failed for {config_path}")
            return False
        print(f"✓ Training complete for {config_path}")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Train QA models with varying Q values"
    )
    parser.add_argument(
        "--q_values",
        type=int,
        nargs="+",
        default=[16, 32, 64, 128, 256],
        help="List of Q values to train (default: 16 32 64 128 256)",
    )
    parser.add_argument(
        "--base_config",
        type=str,
        default="configs/default.yaml",
        help="Base config file to use",
    )
    parser.add_argument(
        "--use_large_data",
        action="store_true",
        help="Use large QA dataset (qa_large_train.json) instead of default",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without executing",
    )

    args = parser.parse_args()

    # Determine data paths
    if args.use_large_data:
        train_data = DATA_DIR / "qa_large_train.json"
        dev_data = DATA_DIR / "qa_large_dev.json"
    else:
        train_data = DATA_DIR / "qa_train.json"
        dev_data = DATA_DIR / "qa_dev.json"

    # Check data exists
    if not train_data.exists():
        print(f"❌ Training data not found: {train_data}")
        print("\nPlease run one of:")
        print("  python scripts/prepare_data.py                    # Standard datasets")
        print("  python scripts/prepare_large_qa_data.py           # Large-scale datasets")
        sys.exit(1)

    if not dev_data.exists():
        print(f"❌ Dev data not found: {dev_data}")
        sys.exit(1)

    print("=" * 70)
    print("TRAIN QA MODELS WITH VARYING Q VALUES")
    print("=" * 70)
    print(f"Q values: {args.q_values}")
    print(f"Base config: {args.base_config}")
    print(f"Training data: {train_data} ({train_data.stat().st_size / 1e6:.1f} MB)")
    print(f"Dev data: {dev_data} ({dev_data.stat().st_size / 1e6:.1f} MB)")
    print(f"Dry run: {args.dry_run}")
    print("=" * 70)
    print()

    # Create configs
    print("Step 1: Creating configs...")
    configs = []
    for q in args.q_values:
        config_path = create_qa_config(
            q_value=q,
            base_config=args.base_config,
        )
        configs.append(config_path)

    # Train models
    print("\nStep 2: Training models...")
    print("=" * 70)

    for i, config_path in enumerate(configs, 1):
        q_val = args.q_values[i - 1]
        print(f"\n[{i}/{len(configs)}] Training Q={q_val}")
        print("-" * 70)

        success = train_qa_model(
            config_path=config_path,
            data_path=str(train_data),
            eval_data_path=str(dev_data),
            dry_run=args.dry_run,
        )

        if not success and not args.dry_run:
            print(f"\n❌ Failed at Q={q_val}. Stopping.")
            break

    print("\n" + "=" * 70)
    print("✓ All training jobs completed (or queued)")
    print("=" * 70)
    print("\nNext step: Evaluate all checkpoints")
    print(f"  python scripts/evaluate_all_checkpoints.py \\")
    print(f"      --eval_data {dev_data} \\")
    print(f"      --stage 2 \\")
    print(f"      --output results/qa_varying_q_results.csv")
    print()


if __name__ == "__main__":
    main()
