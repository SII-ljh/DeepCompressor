"""
Fine-tuning script for financial NER using FinBERT2.

Usage:
    python finetune_ner.py --model_name "valuesimplex-ai-lab/FinBERT2-base"
"""

import argparse
import csv
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    BertForTokenClassification,
    Trainer,
    TrainingArguments,
)

from ner_dataset import NERDataset, NERDataCollator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "valuesimplex-ai-lab/FinBERT2-base"


@dataclass
class NERConfig:
    """Configuration for NER fine-tuning."""
    model_name: str = DEFAULT_MODEL_NAME
    train_data_path: str = "sample_data/train.json"
    test_data_path: str = "sample_data/test.json"
    output_base_dir: str = "ner_models"
    results_csv_path: str = "ner_results.csv"
    num_epochs: int = 20
    train_batch_size: int = 4
    eval_batch_size: int = 8
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    max_length: int = 256
    logging_steps: int = 5
    eval_steps: int = 10
    save_steps: int = 100
    seed: int = 42


def extract_entities_from_ids(label_ids: List[int], id2label: Dict[int, str]) -> List[dict]:
    """Extract entity spans from a BIO label ID sequence."""
    entities = []
    current = None
    for i, lid in enumerate(label_ids):
        if lid == -100:
            continue
        label = id2label.get(lid, "O")
        if label.startswith("B-"):
            if current:
                entities.append(current)
            current = {"type": label[2:], "start": i, "end": i + 1}
        elif label.startswith("I-") and current and current["type"] == label[2:]:
            current["end"] = i + 1
        else:
            if current:
                entities.append(current)
                current = None
    if current:
        entities.append(current)
    return entities


def compute_entity_metrics(
    pred_ids_list: List[List[int]],
    true_ids_list: List[List[int]],
    id2label: Dict[int, str]
) -> Dict[str, float]:
    """Compute entity-level precision, recall, F1."""
    tp, fp, fn = 0, 0, 0
    for preds, trues in zip(pred_ids_list, true_ids_list):
        pred_ents = {
            (e["type"], e["start"], e["end"])
            for e in extract_entities_from_ids(preds, id2label)
        }
        true_ents = {
            (e["type"], e["start"], e["end"])
            for e in extract_entities_from_ids(trues, id2label)
        }
        tp += len(pred_ents & true_ents)
        fp += len(pred_ents - true_ents)
        fn += len(true_ents - pred_ents)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


class NERFinetuner:
    """Financial NER fine-tuning framework based on FinBERT2."""

    def __init__(self, config: NERConfig):
        self.config = config
        self.eval_counter = 0

        model_suffix = config.model_name.split("/")[-1]
        self.experiment_name = f"{model_suffix}_NER"
        self.output_dir = Path(config.output_base_dir) / self.experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Experiment: {self.experiment_name}")

    def setup_reproducibility(self):
        seed = self.config.seed
        torch.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True

    def train(self):
        logger.info("Starting NER fine-tuning pipeline")
        self.setup_reproducibility()

        # Load datasets
        train_dataset = NERDataset(self.config.train_data_path)
        test_dataset = NERDataset(self.config.test_data_path)
        label_list, label2id, id2label = train_dataset.get_label_info()
        self.id2label = id2label

        # Load tokenizer and model
        logger.info(f"Loading model: {self.config.model_name}")
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        model = BertForTokenClassification.from_pretrained(
            self.config.model_name,
            num_labels=len(label_list),
            id2label={str(k): v for k, v in id2label.items()},
            label2id=label2id,
        )

        collator = NERDataCollator(tokenizer, label2id, self.config.max_length)

        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.train_batch_size,
            per_device_eval_batch_size=self.config.eval_batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            save_strategy="steps",
            eval_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            seed=self.config.seed,
            remove_unused_columns=False,
        )
        self.training_args = training_args

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            data_collator=collator,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self._compute_metrics,
        )

        logger.info("Starting training...")
        trainer.train()

        # Save best model
        logger.info(f"Saving model to {self.output_dir}")
        tokenizer.save_pretrained(self.output_dir)
        model.save_pretrained(self.output_dir)
        logger.info("Training completed!")

    def _compute_metrics(self, eval_pred) -> Dict[str, float]:
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)

        metrics = compute_entity_metrics(
            predictions.tolist(), labels.tolist(), self.id2label
        )

        self.eval_counter += 1

        # Log to CSV
        csv_path = Path(self.config.results_csv_path)
        file_exists = csv_path.exists()
        fieldnames = ['experiment', 'eval_step', 'precision', 'recall', 'f1']
        with open(csv_path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                'experiment': self.experiment_name,
                'eval_step': self.eval_counter * self.config.eval_steps,
                'precision': f"{metrics['precision']:.4f}",
                'recall': f"{metrics['recall']:.4f}",
                'f1': f"{metrics['f1']:.4f}",
            })

        return metrics


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Fine-tune FinBERT2 for financial NER'
    )
    parser.add_argument('--model_name', type=str, help='Pretrained model name or path')
    parser.add_argument('--train_data', type=str, help='Path to training data JSON')
    parser.add_argument('--test_data', type=str, help='Path to test data JSON')
    parser.add_argument('--output_dir', type=str, help='Base output directory')
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, help='Learning rate')
    parser.add_argument('--seed', type=int, help='Random seed')
    return parser.parse_args()


def main():
    args = parse_arguments()
    config = NERConfig()

    if args.model_name:
        config.model_name = args.model_name
    if args.train_data:
        config.train_data_path = args.train_data
    if args.test_data:
        config.test_data_path = args.test_data
    if args.output_dir:
        config.output_base_dir = args.output_dir
    if args.num_epochs:
        config.num_epochs = args.num_epochs
    if args.batch_size:
        config.train_batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    if args.seed:
        config.seed = args.seed

    logger.info(f"Using model: {config.model_name}")

    finetuner = NERFinetuner(config)
    finetuner.train()


if __name__ == "__main__":
    main()
