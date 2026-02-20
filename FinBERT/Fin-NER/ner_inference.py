"""
NER inference module for financial entity recognition.

Usage:
    python ner_inference.py <model_path>

    # Or in code:
    engine = NERInferenceEngine("ner_models/FinBERT2-base_NER")
    entities = engine.predict("招商银行2024年净利润285亿元。")
    # [{"text": "招商银行", "label": "ORG", "start": 0, "end": 4}, ...]
"""

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer, BertForTokenClassification

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NERInferenceEngine:
    """Financial NER inference engine based on fine-tuned FinBERT2."""

    def __init__(self, model_path: Union[str, Path], device: Optional[str] = None):
        self.model_path = Path(model_path)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        logger.info(f"Loading NER model from {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = BertForTokenClassification.from_pretrained(model_path).to(self.device)
        self.model.eval()

        # Normalize id2label keys to int
        self.id2label = {
            int(k): v for k, v in self.model.config.id2label.items()
        }
        self._is_fast = getattr(self.tokenizer, "is_fast", False)

        logger.info(f"Model loaded on {self.device}, labels: {list(set(self.id2label.values()))}")

    def predict(self, text: str) -> List[dict]:
        """
        Predict named entities in text.

        Args:
            text: Input text string

        Returns:
            List of entity dicts:
            [{"text": "招商银行", "label": "ORG", "start": 0, "end": 4}, ...]
        """
        if not text.strip():
            return []

        chars = list(text)
        char_labels = self._predict_char_labels(text, chars)

        # Extract entity spans from char-level BIO labels
        entities = []
        current = None
        for i, label in enumerate(char_labels):
            if label.startswith("B-"):
                if current:
                    entities.append(current)
                etype = label[2:]
                current = {"text": chars[i], "label": etype, "start": i, "end": i + 1}
            elif label.startswith("I-") and current and current["label"] == label[2:]:
                current["text"] += chars[i]
                current["end"] = i + 1
            else:
                if current:
                    entities.append(current)
                    current = None
        if current:
            entities.append(current)

        return entities

    def _predict_char_labels(self, text: str, chars: List[str]) -> List[str]:
        """Run model prediction and map token labels back to characters."""
        if self._is_fast:
            return self._predict_fast(chars)
        else:
            return self._predict_slow(text, chars)

    def _predict_fast(self, chars: List[str]) -> List[str]:
        """Predict using fast tokenizer with word_ids() alignment."""
        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**encoding).logits

        pred_ids = torch.argmax(logits, dim=2)[0].cpu().tolist()
        word_ids = encoding.word_ids()

        char_labels = ["O"] * len(chars)
        prev_wid = None
        for token_idx, wid in enumerate(word_ids):
            if wid is None or wid == prev_wid:
                prev_wid = wid
                continue
            if wid < len(char_labels):
                char_labels[wid] = self.id2label.get(pred_ids[token_idx], "O")
            prev_wid = wid

        return char_labels

    def _predict_slow(self, text: str, chars: List[str]) -> List[str]:
        """Predict using slow tokenizer with manual alignment."""
        encoding = self.tokenizer(
            text,
            max_length=512,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            logits = self.model(**encoding).logits

        pred_ids = torch.argmax(logits, dim=2)[0].cpu().tolist()
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])

        char_labels = ["O"] * len(chars)
        char_idx = 0
        for token_idx, token in enumerate(tokens):
            if token in (self.tokenizer.cls_token, self.tokenizer.sep_token,
                         self.tokenizer.pad_token):
                continue
            if token.startswith("##"):
                continue
            if char_idx < len(char_labels):
                char_labels[char_idx] = self.id2label.get(pred_ids[token_idx], "O")
                token_len = len(token.replace("##", ""))
                char_idx += max(token_len, 1)

        return char_labels

    def predict_batch(self, texts: List[str]) -> List[List[dict]]:
        """Predict entities for a batch of texts."""
        return [self.predict(text) for text in texts]

    def get_model_info(self) -> dict:
        """Get model configuration info."""
        return {
            "model_path": str(self.model_path),
            "device": str(self.device),
            "num_labels": len(self.id2label),
            "entity_types": sorted(set(
                v[2:] for v in self.id2label.values() if v.startswith("B-")
            )),
        }


def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "ner_models/FinBERT2-base_NER"

    engine = NERInferenceEngine(model_path)
    print(f"Model info: {json.dumps(engine.get_model_info(), ensure_ascii=False, indent=2)}")

    test_texts = [
        "招商银行2024年第三季度实现净利润285亿元，同比增长5.3%。",
        "宁德时代董事长曾毓群宣布全固态电池将于2027年量产。",
        "央行宣布降准0.5个百分点，释放长期资金约1万亿元。",
        "贵州茅台2024年度营业总收入1738亿元，归母净利润862亿元。",
    ]

    for text in test_texts:
        entities = engine.predict(text)
        print(f"\n{'='*60}")
        print(f"文本: {text}")
        print(f"实体 ({len(entities)}):")
        for ent in entities:
            print(f"  [{ent['label']:>7s}] {ent['text']:<12s}  ({ent['start']}-{ent['end']})")


if __name__ == "__main__":
    main()
