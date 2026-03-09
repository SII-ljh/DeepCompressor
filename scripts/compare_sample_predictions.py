"""Compare sample predictions across different Q values with Qwen baseline.

Shows how the same samples are predicted by:
1. Qwen-0.6B reading full document directly (baseline)
2. Models with different Q values (compressed)

This makes it easy to see the impact of compression ratio on generation quality.

Usage:
    # Compare predictions on 3 samples (includes Qwen baseline)
    python scripts/compare_sample_predictions.py \
        --eval_data data/ntp_train.jsonl \
        --num_samples 3

    # Stage 2 QA comparison
    python scripts/compare_sample_predictions.py \
        --eval_data data/qa_dev.json \
        --stage 2 \
        --num_samples 5

    # Skip baseline for faster evaluation
    python scripts/compare_sample_predictions.py \
        --eval_data data/ntp_train.jsonl \
        --num_samples 3 \
        --no_baseline
"""

import argparse
import logging
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
from transformers import AutoTokenizer

from deep_compressor.config import DeepCompressorConfig
from deep_compressor.data import NTPDataset, PaddingCollator, QADataset
from deep_compressor.model import DeepCompressor

from transformers import AutoModelForCausalLM

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def find_all_checkpoints(base_dir: str = "outputs") -> List[Path]:
    """Find all checkpoint directories."""
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    checkpoints = []
    for checkpoint_dir in base_path.iterdir():
        if checkpoint_dir.is_dir():
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


def load_model(checkpoint_path: Path, config_path: Path,
               device: torch.device) -> DeepCompressor:
    """Load model from checkpoint."""
    config = DeepCompressorConfig.from_yaml(str(config_path))
    model = DeepCompressor(config).to(device)

    weights_path = checkpoint_path / "checkpoint-final" / "trainable_weights.pt"
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(weights, strict=False)
    model.eval()

    return model


@torch.no_grad()
def predict_ntp_samples(model, samples, tokenizer, device, max_gen=50):
    """Generate predictions for NTP samples."""
    predictions = []

    for sample in samples:
        doc_ids = sample["doc_input_ids"].unsqueeze(0).to(device)
        doc_mask = sample["doc_attention_mask"].unsqueeze(0).to(device)
        segment_ids = sample["segment_ids"].unsqueeze(0).to(device)

        # Encode and compress
        byte_array = model.encode_document(doc_ids, doc_mask)
        B = doc_ids.shape[0]
        zero_pooled = torch.zeros(B, model.config.qwen.hidden_size, device=device)
        queries = model.query_init(zero_pooled)
        latent = model.compress(queries, byte_array, byte_mask=doc_mask)
        prefix_embeds = model.up_mlp(latent)

        # Generate
        embed_layer = model.qwen.get_input_embeddings()
        dummy_embeds = embed_layer(segment_ids[:, :1])
        prefix_embeds = prefix_embeds.to(dtype=dummy_embeds.dtype)

        prefix_len = prefix_embeds.shape[1]
        prefix_mask = torch.ones(B, prefix_len, device=device)

        gen_ids = model.qwen.generate(
            inputs_embeds=prefix_embeds,
            attention_mask=prefix_mask,
            max_new_tokens=min(max_gen, segment_ids.shape[1]),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        pred_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        gold_text = tokenizer.decode(segment_ids[0, :max_gen], skip_special_tokens=True)

        predictions.append({
            "prediction": pred_text,
            "gold": gold_text,
        })

    return predictions


@torch.no_grad()
def predict_qa_samples(model, samples, tokenizer, device, max_new_tokens=64):
    """Generate predictions for QA samples."""
    predictions = []

    for sample in samples:
        doc_ids = sample["doc_input_ids"].unsqueeze(0).to(device)
        doc_mask = sample["doc_attention_mask"].unsqueeze(0).to(device)
        q_ids = sample["q_input_ids"].unsqueeze(0).to(device)
        q_mask = sample["q_attention_mask"].unsqueeze(0).to(device)

        # Encode and compress
        byte_array = model.encode_document(doc_ids, doc_mask)
        queries = model.encode_question(q_ids, q_mask)
        latent = model.compress(queries, byte_array, byte_mask=doc_mask)
        prefix_embeds = model.up_mlp(latent)

        # Generate
        gen_ids = model.generate_answer(
            prefix_embeds, q_ids, q_mask,
            tokenizer=tokenizer, max_new_tokens=max_new_tokens
        )

        pred_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

        predictions.append({
            "question": sample.get("question_text", ""),
            "prediction": pred_text,
            "gold": sample["answer_text"],
        })

    return predictions


@torch.no_grad()
def predict_qwen_baseline_ntp(qwen_model, samples, tokenizer, device, max_gen=50):
    """Generate predictions using Qwen baseline (reading full document)."""
    predictions = []

    for sample in samples:
        doc_ids = sample["doc_input_ids"].unsqueeze(0).to(device)
        doc_mask = sample["doc_attention_mask"].unsqueeze(0).to(device)
        segment_ids = sample["segment_ids"].unsqueeze(0).to(device)

        # Qwen reads full document directly
        gen_ids = qwen_model.generate(
            input_ids=doc_ids,
            attention_mask=doc_mask,
            max_new_tokens=min(max_gen, segment_ids.shape[1]),
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Extract only generated tokens (strip input)
        gen_tokens = gen_ids[0, doc_ids.shape[1]:]
        pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)
        gold_text = tokenizer.decode(segment_ids[0, :max_gen], skip_special_tokens=True)

        predictions.append({
            "prediction": pred_text,
            "gold": gold_text,
        })

    return predictions


@torch.no_grad()
def predict_qwen_baseline_qa(qwen_model, samples, tokenizer, device, max_new_tokens=64):
    """Generate predictions using Qwen baseline (reading full document + question)."""
    predictions = []

    for sample in samples:
        doc_ids = sample["doc_input_ids"].unsqueeze(0).to(device)
        doc_mask = sample["doc_attention_mask"].unsqueeze(0).to(device)
        q_ids = sample["q_input_ids"].unsqueeze(0).to(device)
        q_mask = sample["q_attention_mask"].unsqueeze(0).to(device)

        # Concatenate document + question
        input_ids = torch.cat([doc_ids, q_ids], dim=1)
        attention_mask = torch.cat([doc_mask, q_mask], dim=1)

        # Generate answer
        gen_ids = qwen_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        # Extract only generated tokens
        gen_tokens = gen_ids[0, input_ids.shape[1]:]
        pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        predictions.append({
            "question": sample.get("question_text", ""),
            "prediction": pred_text,
            "gold": sample["answer_text"],
        })

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Compare sample predictions across different Q values")
    parser.add_argument("--eval_data", type=str, required=True,
                        help="Path to evaluation data")
    parser.add_argument("--stage", type=int, default=1, choices=[1, 2],
                        help="Evaluation stage (1=NTP, 2=QA)")
    parser.add_argument("--num_samples", type=int, default=3,
                        help="Number of samples to compare")
    parser.add_argument("--max_gen_tokens", type=int, default=50,
                        help="Max tokens to generate per sample")
    parser.add_argument("--no_baseline", action="store_true",
                        help="Skip Qwen baseline comparison")
    args = parser.parse_args()

    # Find all checkpoints
    checkpoint_dirs = find_all_checkpoints()
    if not checkpoint_dirs:
        logger.error("No checkpoints found")
        return

    # Sort by Q value
    checkpoint_dirs.sort(key=extract_q_value)

    logger.info(f"Found {len(checkpoint_dirs)} checkpoints")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "models/Qwen3-0.6B",
        trust_remote_code=True,
        fix_mistral_regex=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare data
    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)

    if args.stage == 1:
        dataset = NTPDataset(
            args.eval_data, tokenizer,
            max_doc_tokens=512,
            segment_len=256
        )
    else:
        dataset = QADataset(
            args.eval_data, tokenizer,
            max_doc_tokens=512,
            max_question_tokens=256,
            max_answer_tokens=512
        )

    # Take validation subset
    if args.stage == 1:
        n_total = len(dataset)
        n_train = n_total - min(5000, n_total // 10)
        val_subset = Subset(dataset, list(range(n_train, n_total)))
    else:
        val_subset = dataset

    # Select fixed samples
    sample_indices = list(range(min(args.num_samples, len(val_subset))))
    samples = [val_subset[i] for i in sample_indices]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load Qwen baseline model (optional)
    baseline_predictions = None
    if not args.no_baseline:
        logger.info("\nLoading Qwen baseline model (reading full document)...")
        qwen_baseline = AutoModelForCausalLM.from_pretrained(
            "models/Qwen3-0.6B",
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True
        )
        qwen_baseline.eval()

        logger.info("Generating Qwen baseline predictions...")
        if args.stage == 1:
            baseline_predictions = predict_qwen_baseline_ntp(
                qwen_baseline, samples, tokenizer, device, max_gen=args.max_gen_tokens
            )
        else:
            baseline_predictions = predict_qwen_baseline_qa(
                qwen_baseline, samples, tokenizer, device, max_new_tokens=args.max_gen_tokens
            )

        # Free memory
        del qwen_baseline
        torch.cuda.empty_cache()
        logger.info("Baseline predictions complete\n")

    # Evaluate each checkpoint on the same samples
    all_predictions = {}

    for checkpoint_dir in checkpoint_dirs:
        q_value = extract_q_value(checkpoint_dir)
        logger.info(f"\nLoading Q={q_value} model...")

        # Find config
        config_name = checkpoint_dir.name
        if "stage1_q" in config_name:
            config_path = Path("configs") / f"{config_name}.yaml"
        else:
            config_path = Path("configs") / "default.yaml"

        if not config_path.exists():
            logger.warning(f"Config not found: {config_path}, skipping")
            continue

        # Load model
        model = load_model(checkpoint_dir, config_path, device)

        # Generate predictions
        if args.stage == 1:
            predictions = predict_ntp_samples(
                model, samples, tokenizer, device, max_gen=args.max_gen_tokens
            )
        else:
            predictions = predict_qa_samples(
                model, samples, tokenizer, device, max_new_tokens=args.max_gen_tokens
            )

        all_predictions[q_value] = predictions

    # Display comparison
    print("\n" + "="*100)
    print("  Sample Prediction Comparison: Qwen Baseline vs Compressed Models")
    print("="*100 + "\n")

    for sample_idx in range(len(samples)):
        print("━"*100)
        print(f"Sample {sample_idx + 1}")
        print("━"*100)

        if args.stage == 2 and "question" in all_predictions[list(all_predictions.keys())[0]][sample_idx]:
            question = all_predictions[list(all_predictions.keys())[0]][sample_idx]["question"]
            print(f"\n【Question】: {question}\n")

        print(f"【Gold Answer】:")
        gold = all_predictions[list(all_predictions.keys())[0]][sample_idx]["gold"]
        print(f"  {gold}\n")

        # Show Qwen baseline first (if available)
        if baseline_predictions:
            baseline_pred = baseline_predictions[sample_idx]["prediction"]
            print(f"【Qwen-0.6B Baseline (Full Document)】:")
            print(f"  {baseline_pred}")
            print()
            print("-"*100)
            print()

        # Show compressed models
        for q_value in sorted(all_predictions.keys()):
            pred = all_predictions[q_value][sample_idx]["prediction"]
            print(f"【Q={q_value:3d} Compressed】:")
            print(f"  {pred}")
            print()

        print("="*100 + "\n")


if __name__ == "__main__":
    main()
