"""Evaluate all trained checkpoints with comprehensive metrics and unified output.

Automatically discovers all checkpoint directories, evaluates models,
and presents results in a unified comparison table with sample predictions.

Features:
  - Evaluates all models first, then displays unified comparison table
  - Shows comprehensive metrics: perplexity, loss, token accuracy, top-5 accuracy
  - Includes direct_qwen baseline (Stage 1 only) for performance comparison
  - Displays sample predictions for each model with doc preview, prediction, and gold
  - Calculates quality retention rate (compressed vs baseline)

Metrics Explained:
  - perplexity: Prediction uncertainty (lower is better)
  - loss: Cross-entropy loss (lower is better)
  - token_accuracy: Top-1 token prediction accuracy (higher is better)
  - top5_accuracy: Correct token in top-5 predictions (higher is better)
  - Retention: Quality preserved after compression (100% = baseline performance)

Usage:
    # Evaluate all checkpoints on Stage 1 NTP validation data
    # (includes direct_qwen baseline + comprehensive metrics)
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

    # Collect more sample predictions (default: 3)
    python scripts/evaluate_all_checkpoints.py \
        --eval_data data/ntp_train.jsonl \
        --stage 1 \
        --show_samples 5

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
                              accelerator: Accelerator, tokenizer,
                              collect_samples: int = 3,
                              max_gen_tokens: int = 50) -> Dict:
    """Evaluate direct Qwen (no compression) on NTP task as upper bound.

    Args:
        eval_loader: DataLoader for NTP validation data (already prepared)
        qwen_model: Raw Qwen model (not wrapped in DeepCompressor)
        accelerator: Accelerator instance
        tokenizer: Tokenizer for generating sample predictions
        collect_samples: Number of sample predictions to collect
        max_gen_tokens: Maximum tokens to generate per sample
    Returns:
        {
            "perplexity": float,
            "loss": float,
            "token_accuracy": float,
            "top5_accuracy": float,
            "samples": [{"prediction": str, "gold": str, "doc_preview": str}, ...]
        }
    """
    logger.info("Evaluating direct_qwen baseline (upper bound - no compression)")
    qwen_model.eval()

    total_loss = 0.0
    total_samples = 0
    total_correct_tokens = 0
    total_top5_correct = 0
    total_tokens = 0
    samples_collected = []

    for batch_idx, batch in enumerate(eval_loader):
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

        # Calculate token accuracy (only on main process)
        if accelerator.is_main_process:
            logits = outputs.logits  # (B, seq_len, vocab_size)

            # Find where segment starts in the concatenated sequence
            doc_len = doc_ids.shape[1]
            segment_logits = logits[:, doc_len:-1, :].contiguous()  # Shift for next-token
            shift_labels = segment_labels[:, 1:].contiguous()

            valid_mask = (shift_labels != -100)
            if valid_mask.any():
                # Top-1 accuracy
                predictions = segment_logits.argmax(dim=-1)
                correct = (predictions == shift_labels) & valid_mask
                total_correct_tokens += correct.sum().item()

                # Top-5 accuracy
                top5_preds = segment_logits.topk(5, dim=-1).indices
                shift_labels_expanded = shift_labels.unsqueeze(-1).expand_as(top5_preds)
                top5_correct = ((top5_preds == shift_labels_expanded).any(dim=-1)) & valid_mask
                total_top5_correct += top5_correct.sum().item()

                total_tokens += valid_mask.sum().item()

        # Collect sample predictions (only on main process)
        if (tokenizer is not None and len(samples_collected) < collect_samples
                and accelerator.is_main_process):
            try:
                # Take first sample from batch
                doc_ids_sample = doc_ids[:1]
                doc_mask_sample = doc_mask[:1]

                # Generate continuation
                gen_ids = qwen_model.generate(
                    input_ids=doc_ids_sample,
                    attention_mask=doc_mask_sample,
                    max_new_tokens=max_gen_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                # Decode (strip input portion)
                gen_ids_only = gen_ids[:, doc_ids_sample.shape[1]:]
                pred_text = tokenizer.decode(gen_ids_only[0], skip_special_tokens=True)
                gold_text = tokenizer.decode(segment_ids[0, :max_gen_tokens], skip_special_tokens=True)
                doc_preview = tokenizer.decode(doc_ids[0, :50], skip_special_tokens=True)

                samples_collected.append({
                    "prediction": pred_text,
                    "gold": gold_text,
                    "doc_preview": doc_preview,
                })
            except Exception as e:
                pass  # Skip failed samples

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

    # Calculate accuracies
    token_acc = total_correct_tokens / max(total_tokens, 1) if total_tokens > 0 else 0.0
    top5_acc = total_top5_correct / max(total_tokens, 1) if total_tokens > 0 else 0.0

    return {
        "perplexity": perplexity,
        "loss": avg_loss,
        "token_accuracy": token_acc,
        "top5_accuracy": top5_acc,
        "samples": samples_collected,
    }


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
                        collect_samples: int = 3) -> Dict:
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
                               tokenizer=tokenizer, collect_samples=collect_samples)

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
            show_samples=collect_samples,
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

    # Get metric names, excluding 'samples'
    all_metrics = results[0]["metrics"].keys()
    metric_names = [m for m in all_metrics if m != "samples"]

    fieldnames = ["checkpoint", "q_value"] + metric_names

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for result in results:
            q_val = result["q_value"]
            # Format q_value for readability
            q_display = "baseline" if q_val == 999999 else str(q_val)

            # Build row, excluding samples
            metrics_without_samples = {k: v for k, v in result["metrics"].items()
                                       if k != "samples"}

            row = {
                "checkpoint": result["checkpoint"],
                "q_value": q_display,
                **metrics_without_samples,
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
            direct_qwen_metrics = evaluate_direct_qwen_ntp(
                eval_loader, qwen_model, accelerator, tokenizer,
                collect_samples=args.show_samples)

            # Add to results
            results.append({
                "checkpoint": "direct_qwen (baseline)",
                "q_value": 999999,  # Sort to top
                "metrics": direct_qwen_metrics,
            })

            logger.info(f"✓ Direct Qwen baseline evaluation complete")

            # Free memory
            del qwen_model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            logger.error(f"Failed to evaluate direct_qwen baseline: {e}", exc_info=True)

    # Evaluate each checkpoint (collect results quietly)
    for checkpoint_dir in checkpoint_dirs:
        logger.info(f"\nEvaluating: {checkpoint_dir.name}...")

        try:
            metrics = evaluate_checkpoint(
                checkpoint_dir=checkpoint_dir,
                eval_data_path=args.eval_data,
                stage=args.stage,
                accelerator=accelerator,
                tokenizer=tokenizer,
                max_eval_samples=args.max_eval_samples,
                collect_samples=args.show_samples,
            )

            if metrics:
                q_value = extract_q_value(checkpoint_dir)
                results.append({
                    "checkpoint": checkpoint_dir.name,
                    "q_value": q_value,
                    "metrics": metrics,
                })
                logger.info(f"✓ {checkpoint_dir.name} evaluation complete")

        except Exception as e:
            logger.error(f"✗ Failed to evaluate {checkpoint_dir.name}: {e}")

    # Print comparison table
    if results:
        logger.info("\n" + "="*100)
        logger.info("EVALUATION RESULTS - COMPARISON TABLE")
        logger.info("="*100)

        # Sort: direct_qwen first (q_value=999999), then by Q value ascending
        results.sort(key=lambda x: (0 if x["q_value"] == 999999 else 1, x["q_value"]))

        # Get metric names (exclude 'samples' from table)
        all_metrics = results[0]["metrics"].keys()
        metric_names = [m for m in all_metrics if m != "samples"]

        # Print header
        header = f"{'Model':<20} | " + " | ".join(f"{m:<14}" for m in metric_names)
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
                model_name = "Qwen (no compress)"
            else:
                model_name = f"Q={q}"

            values = [result["metrics"].get(m, float('nan')) for m in metric_names]
            row = f"{model_name:<20} | " + " | ".join(f"{v:>14.6f}" for v in values)

            # Add retention calculation (only for compressed models)
            if q != 999999 and baseline_loss is not None:
                model_loss = result["metrics"].get("loss")
                if model_loss is not None:
                    retention = baseline_loss / model_loss if model_loss > 0 else 0.0
                    retention_pct = min(retention * 100, 100.0)
                    row += f" | {retention_pct:>9.1f}%"
                else:
                    row += " | " + " "*10 + "N/A"
            elif direct_qwen_metrics:
                row += " | " + " "*10 + "—"

            logger.info(row)

        logger.info("="*100)

        # Print metric explanations
        logger.info("\nMetric Explanations:")
        logger.info("  • perplexity: Lower is better (measures prediction uncertainty)")
        logger.info("  • loss: Lower is better (cross-entropy loss)")
        logger.info("  • token_accuracy: Higher is better (top-1 token prediction accuracy)")
        logger.info("  • top5_accuracy: Higher is better (correct token in top-5 predictions)")
        logger.info("  • Retention: Quality preserved after compression (100% = same as baseline)")

        logger.info("\n")

        # Now print sample predictions for each model
        logger.info("\n" + "="*100)
        logger.info("SAMPLE PREDICTIONS")
        logger.info("="*100)

        for result in results:
            checkpoint_name = result["checkpoint"]
            q = result["q_value"]
            samples = result["metrics"].get("samples", [])

            if q == 999999:
                model_name = "Direct Qwen (No Compression - Baseline)"
            else:
                model_name = f"Model: Q={q} ({checkpoint_name})"

            logger.info(f"\n{'─'*100}")
            logger.info(f"  {model_name}")
            logger.info(f"{'─'*100}")

            if not samples:
                logger.info("  (No samples collected)")
                continue

            for i, sample in enumerate(samples, 1):
                logger.info(f"\n  [Sample {i}/{len(samples)}]")
                logger.info(f"  Doc Preview: {sample['doc_preview'][:80]}...")
                logger.info(f"  Prediction:  {sample['prediction'][:200]}")
                logger.info(f"  Gold:        {sample['gold'][:200]}")

                if i < len(samples):
                    logger.info("")

        logger.info("\n" + "="*100 + "\n")

    # Save to CSV if requested
    if args.output:
        save_results(results, args.output)


if __name__ == "__main__":
    main()
