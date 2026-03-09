"""Compare question-guided vs unguided compression on Stage 1 NTP task.

Evaluates two Stage 1 checkpoints (one trained with questions, one without)
on the same NTP eval set and QA zero-shot task to measure compression quality.

Usage:
    python scripts/compare_guided_vs_unguided.py \
        --unguided outputs/stage1_q128/checkpoint-final \
        --guided outputs/stage1_guided_q128/checkpoint-final \
        --eval_data data/ntp_train_512.jsonl \
        --qa_data data/qa_dev.json \
        --num_samples 50
"""

import argparse
import json
import logging
import re
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


def load_model_from_checkpoint(checkpoint_dir: str, config: DeepCompressorConfig) -> DeepCompressor:
    """Load DeepCompressor model from checkpoint."""
    model = DeepCompressor(config)
    weights_path = Path(checkpoint_dir) / "trainable_weights.pt"

    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=False)
    logger.info(f"Loaded model from {checkpoint_dir}")
    return model


def analyze_sample_quality(samples: List[str]) -> Dict[str, float]:
    """Analyze sample quality by checking for presence of factual information.

    Checks for:
    - Numbers (dates, percentages, amounts)
    - Entity names (companies, people, places)
    - Financial terms

    Returns dict with quality metrics.
    """
    has_numbers = 0
    has_entities = 0
    has_financial_terms = 0

    # Financial term patterns (Chinese)
    financial_patterns = [
        r"收入|利润|资产|负债|股东|投资|融资|营收|净利|毛利",
        r"亿元|万元|百分|增长|下降|同比|环比",
        r"公司|企业|集团|股份|有限",
    ]

    for sample in samples:
        # Check for numbers
        if re.search(r"\d+", sample):
            has_numbers += 1

        # Check for company/entity names (common patterns in Chinese)
        if re.search(r"[A-Z]{2,}|[\u4e00-\u9fa5]{2,}公司|[\u4e00-\u9fa5]{2,}集团", sample):
            has_entities += 1

        # Check for financial terms
        if any(re.search(pat, sample) for pat in financial_patterns):
            has_financial_terms += 1

    total = len(samples)
    return {
        "pct_with_numbers": 100 * has_numbers / total if total > 0 else 0,
        "pct_with_entities": 100 * has_entities / total if total > 0 else 0,
        "pct_with_financial_terms": 100 * has_financial_terms / total if total > 0 else 0,
        "overall_factual_quality": 100 * (has_numbers + has_entities + has_financial_terms) / (3 * total) if total > 0 else 0,
    }


def compare_models(
    unguided_checkpoint: str,
    guided_checkpoint: str,
    config: DeepCompressorConfig,
    eval_data: str,
    qa_data: str = None,
    num_samples: int = 50,
):
    """Compare guided vs unguided models."""
    accelerator = Accelerator()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.qwen.model_name_or_path,
        trust_remote_code=True,
        fix_mistral_regex=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load models
    logger.info("Loading unguided model...")
    unguided_model = load_model_from_checkpoint(unguided_checkpoint, config)

    logger.info("Loading guided model...")
    guided_model = load_model_from_checkpoint(guided_checkpoint, config)

    # Prepare models
    unguided_model = accelerator.prepare(unguided_model)
    guided_model = accelerator.prepare(guided_model)

    unguided_model.eval()
    guided_model.eval()

    # ── NTP Evaluation ──
    logger.info("\n" + "="*80)
    logger.info("NTP PERPLEXITY EVALUATION")
    logger.info("="*80)

    ntp_dataset = NTPDataset(
        eval_data, tokenizer,
        max_doc_tokens=config.qwen.max_doc_tokens,
        segment_len=config.training.ntp_segment_len,
        use_questions=False  # Evaluate without questions
    )

    # Limit to num_samples
    if num_samples > 0 and num_samples < len(ntp_dataset):
        ntp_dataset = Subset(ntp_dataset, list(range(num_samples)))

    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)
    ntp_loader = DataLoader(ntp_dataset, batch_size=4, shuffle=False, collate_fn=collator)
    ntp_loader = accelerator.prepare(ntp_loader)

    logger.info(f"\nEvaluating UNGUIDED model on NTP...")
    unguided_ntp = evaluate_ntp(unguided_model, ntp_loader, accelerator, tokenizer=tokenizer, show_sample=True)

    logger.info(f"\nEvaluating GUIDED model on NTP...")
    guided_ntp = evaluate_ntp(guided_model, ntp_loader, accelerator, tokenizer=tokenizer, show_sample=True)

    # ── QA Zero-shot Evaluation (optional) ──
    qa_metrics = None
    if qa_data and Path(qa_data).exists():
        logger.info("\n" + "="*80)
        logger.info("ZERO-SHOT QA EVALUATION (Stage 1 only, no QA fine-tuning)")
        logger.info("="*80)

        qa_dataset = QADataset(
            qa_data, tokenizer,
            max_doc_tokens=config.qwen.max_doc_tokens,
            max_question_tokens=config.qwen.max_question_tokens,
            max_answer_tokens=config.qwen.max_answer_tokens,
        )

        if num_samples > 0 and num_samples < len(qa_dataset):
            qa_dataset = Subset(qa_dataset, list(range(num_samples)))

        qa_loader = DataLoader(qa_dataset, batch_size=4, shuffle=False, collate_fn=collator)
        qa_loader = accelerator.prepare(qa_loader)

        logger.info(f"\nEvaluating UNGUIDED model on zero-shot QA...")
        unguided_qa = evaluate_qa(
            unguided_model, qa_loader, tokenizer, accelerator,
            max_new_tokens=config.qwen.max_answer_tokens,
            show_samples=3
        )

        logger.info(f"\nEvaluating GUIDED model on zero-shot QA...")
        guided_qa = evaluate_qa(
            guided_model, qa_loader, tokenizer, accelerator,
            max_new_tokens=config.qwen.max_answer_tokens,
            show_samples=3
        )

        qa_metrics = {
            "unguided": unguided_qa,
            "guided": guided_qa,
        }

    # ── Comparison Summary ──
    if accelerator.is_main_process:
        logger.info("\n" + "="*80)
        logger.info("COMPARISON SUMMARY")
        logger.info("="*80)

        logger.info("\n📊 NTP Perplexity (lower is better):")
        logger.info(f"  Unguided: {unguided_ntp['perplexity']:.2f}  |  Loss: {unguided_ntp['loss']:.4f}")
        logger.info(f"  Guided:   {guided_ntp['perplexity']:.2f}  |  Loss: {guided_ntp['loss']:.4f}")

        ppl_improvement = (unguided_ntp['perplexity'] - guided_ntp['perplexity']) / unguided_ntp['perplexity'] * 100
        logger.info(f"  → Improvement: {ppl_improvement:+.1f}%")

        if qa_metrics:
            logger.info("\n📊 Zero-shot QA (without Stage 2 fine-tuning):")
            logger.info(f"  Unguided: EM={unguided_qa['exact_match']:.2%}  |  F1={unguided_qa['f1']:.4f}")
            logger.info(f"  Guided:   EM={guided_qa['exact_match']:.2%}  |  F1={guided_qa['f1']:.4f}")

            em_improvement = (guided_qa['exact_match'] - unguided_qa['exact_match']) * 100
            f1_improvement = (guided_qa['f1'] - unguided_qa['f1']) * 100
            logger.info(f"  → EM improvement: {em_improvement:+.1f} points")
            logger.info(f"  → F1 improvement: {f1_improvement:+.4f} points")

        logger.info("\n" + "="*80)

    return {
        "ntp": {"unguided": unguided_ntp, "guided": guided_ntp},
        "qa": qa_metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare guided vs unguided Stage 1 models"
    )
    parser.add_argument("--unguided", type=str, required=True,
                        help="Path to unguided checkpoint directory")
    parser.add_argument("--guided", type=str, required=True,
                        help="Path to guided checkpoint directory")
    parser.add_argument("--config", type=str, default="configs/stage1_q128.yaml",
                        help="Config file (should match checkpoint architecture)")
    parser.add_argument("--eval_data", type=str, required=True,
                        help="NTP evaluation data (JSONL)")
    parser.add_argument("--qa_data", type=str, default=None,
                        help="Optional QA data for zero-shot evaluation")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of samples to evaluate (0 = all)")

    args = parser.parse_args()

    config = DeepCompressorConfig.from_yaml(args.config)

    compare_models(
        unguided_checkpoint=args.unguided,
        guided_checkpoint=args.guided,
        config=config,
        eval_data=args.eval_data,
        qa_data=args.qa_data,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()
