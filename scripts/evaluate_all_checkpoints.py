"""Evaluate all trained checkpoints in outputs/ directory.

Automatically discovers all checkpoint directories, loads models,
and evaluates them on the test set. Shows comprehensive metrics
and sample predictions.

For Stage 1 (NTP), also evaluates direct_qwen baseline (Qwen reading
full document without compression) as an upper bound for comparison.

Usage:
    # Evaluate all checkpoints on Stage 1 NTP validation data
    # (includes direct_qwen baseline comparison)
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
import sys
from pathlib import Path
from typing import Dict, List

# Add project root to Python path
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, Subset
from transformers import AutoModelForCausalLM, AutoTokenizer

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


@torch.no_grad()
def evaluate_direct_qwen_ntp(eval_loader: DataLoader, qwen_model,
                              accelerator: Accelerator) -> Dict[str, float]:
    """Evaluate direct Qwen (no compression) on NTP task as upper bound.

    Args:
        eval_loader: DataLoader for NTP validation data (already prepared)
        qwen_model: Raw Qwen model (not wrapped in DeepCompressor)
        accelerator: Accelerator instance
    Returns:
        {"perplexity": float, "loss": float}
    """
    logger.info("Evaluating direct_qwen baseline (upper bound - no compression)")
    qwen_model.eval()

    total_loss = 0.0
    total_samples = 0

    for batch in eval_loader:
        # For direct Qwen, concatenate doc + segment as full context
        doc_ids = batch["doc_input_ids"]
        doc_mask = batch["doc_attention_mask"]
        segment_ids = batch["segment_ids"]
        segment_mask = batch["segment_attention_mask"]
        segment_labels = batch["segment_labels"]

        # Concatenate doc + segment
        input_ids = torch.cat([doc_ids, segment_ids], dim=1)
        attention_mask = torch.cat([doc_mask, segment_mask], dim=1)

        # Labels: -100 for doc portion, real labels for segment portion
        doc_labels = torch.full_like(doc_ids, -100)
        labels = torch.cat([doc_labels, segment_labels], dim=1)

        # Forward pass (Qwen will shift labels internally)
        outputs = qwen_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            use_cache=False,
        )

        loss_val = outputs.loss.detach()
        bs = doc_ids.shape[0]

        total_loss += loss_val * bs
        total_samples += bs

    # Gather across processes
    stats = torch.tensor([total_loss, float(total_samples)],
                         device=accelerator.device)
    stats = accelerator.gather(stats)
    if stats.dim() > 1:
        stats = stats.sum(dim=0)
    gathered_loss = stats[0].item()
    gathered_samples = stats[1].item()

    avg_loss = gathered_loss / max(gathered_samples, 1)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    return {"perplexity": perplexity, "loss": avg_loss}


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
            q_val = result["q_value"]
            # Format q_value for readability
            q_display = "baseline" if q_val == 999999 else str(q_val)

            row = {
                "checkpoint": result["checkpoint"],
                "q_value": q_display,
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

    # Results list
    results = []
    direct_qwen_metrics = None

    # Evaluate direct_qwen baseline first (only for Stage 1 NTP)
    if args.stage == 1:
        logger.info("\n" + "="*80)
        logger.info("Evaluating BASELINE: Direct Qwen (No Compression)")
        logger.info("="*80 + "\n")

        try:
            # Load Qwen model
            logger.info("Loading Qwen3-0.6B model...")
            qwen_model = AutoModelForCausalLM.from_pretrained(
                "models/Qwen3-0.6B",
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )
            qwen_model.eval()
            for p in qwen_model.parameters():
                p.requires_grad = False
            qwen_model = accelerator.prepare(qwen_model)

            # Prepare eval data (use same config as first checkpoint for data params)
            first_config_path = None
            for checkpoint_dir in checkpoint_dirs:
                config_name = checkpoint_dir.name
                if "stage1_q" in config_name:
                    first_config_path = Path("configs") / f"{config_name}.yaml"
                    if first_config_path.exists():
                        break
            if not first_config_path or not first_config_path.exists():
                first_config_path = Path("configs") / "default.yaml"

            config = DeepCompressorConfig.from_yaml(str(first_config_path))

            # Prepare data
            collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
            num_workers = 0 if accelerator.device.type == "mps" else 2
            pin_memory = accelerator.device.type == "cuda"

            dataset = NTPDataset(
                args.eval_data, tokenizer,
                max_doc_tokens=config.qwen.max_doc_tokens,
                segment_len=config.training.ntp_segment_len,
            )

            # Take validation split (last 10%)
            n_total = len(dataset)
            n_val = min(5000, n_total // 10) if n_total > 10 else n_total
            n_train = n_total - n_val
            val_indices = list(range(n_train, n_total))
            val_subset = Subset(dataset, val_indices)

            if args.max_eval_samples > 0:
                val_subset = Subset(val_subset, list(range(min(args.max_eval_samples, len(val_subset)))))

            eval_loader = DataLoader(
                val_subset, batch_size=config.training.batch_size,
                shuffle=False, collate_fn=collator,
                num_workers=num_workers, pin_memory=pin_memory,
            )
            eval_loader = accelerator.prepare(eval_loader)

            logger.info(f"Evaluating direct_qwen on {len(val_subset)} NTP samples")
            direct_qwen_metrics = evaluate_direct_qwen_ntp(eval_loader, qwen_model, accelerator)

            # Add to results
            results.append({
                "checkpoint": "direct_qwen (baseline)",
                "q_value": 999999,  # Sort to top
                "metrics": direct_qwen_metrics,
            })

            logger.info(f"\n{'='*80}")
            logger.info(f"Direct Qwen Baseline Results (Upper Bound)")
            logger.info(f"{'='*80}")
            for key, value in direct_qwen_metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {key:20s}: {value:.4f}")
            logger.info(f"{'='*80}\n")

            # Free memory
            del qwen_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            logger.error(f"Failed to evaluate direct_qwen baseline: {e}", exc_info=True)

    # Evaluate each checkpoint

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

        # Sort: direct_qwen first (q_value=999999), then by Q value ascending
        results.sort(key=lambda x: (0 if x["q_value"] == 999999 else 1, x["q_value"]))

        # Print header
        metric_names = list(results[0]["metrics"].keys())
        header = f"{'Model':<25} | " + " | ".join(f"{m:<12}" for m in metric_names)
        if direct_qwen_metrics:
            header += " | Retention"
        logger.info(header)
        logger.info("-" * len(header))

        # Get baseline metrics for retention calculation
        baseline_loss = direct_qwen_metrics.get("loss") if direct_qwen_metrics else None

        # Print rows
        for result in results:
            checkpoint_name = result["checkpoint"]
            q = result["q_value"]

            # Format model name
            if q == 999999:
                model_name = "Direct Qwen (baseline)"
            else:
                model_name = f"Q={q}"

            values = [result["metrics"].get(m, float('nan')) for m in metric_names]
            row = f"{model_name:<25} | " + " | ".join(f"{v:>12.4f}" for v in values)

            # Add retention calculation (only for compressed models, based on loss)
            if q != 999999 and baseline_loss is not None:
                model_loss = result["metrics"].get("loss")
                if model_loss is not None:
                    # Lower loss is better; retention = baseline_loss / model_loss
                    # If model achieves same loss as baseline, retention = 100%
                    # If model has 2x higher loss, retention = 50%
                    retention = baseline_loss / model_loss if model_loss > 0 else 0.0
                    retention_pct = min(retention * 100, 100.0)  # Cap at 100%
                    row += f" | {retention_pct:>9.1f}%"
                else:
                    row += " | " + " "*10 + "N/A"
            elif direct_qwen_metrics:
                row += " | " + " "*10 + "—"

            logger.info(row)

        logger.info("="*80)

        # Print summary statistics
        if direct_qwen_metrics and args.stage == 1:
            logger.info("\nSummary:")
            logger.info(f"  - Direct Qwen reads full document ({list(results[0]['metrics'].keys())[0]} tokens)")
            logger.info(f"  - Compressed models use Q prefix vectors")
            logger.info(f"  - Retention = (baseline_loss / model_loss) × 100%")

        logger.info("\n")

    # Save to CSV if requested
    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
