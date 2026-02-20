"""
NER dataset utilities for financial named entity recognition.

Supports JSON format with span-level entity annotations.
Converts to BIO tags and handles subword alignment for BERT tokenization.

Data format:
[
    {
        "text": "招商银行2024年净利润285亿元",
        "entities": [
            {"start": 0, "end": 4, "label": "ORG", "text": "招商银行"},
            ...
        ]
    },
    ...
]
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# Financial NER entity types
ENTITY_TYPES = ["ORG", "PER", "METRIC", "VALUE", "DATE", "EVENT", "PRODUCT"]


def build_label_list(entity_types: Optional[List[str]] = None) -> List[str]:
    """Build BIO label list from entity types."""
    types = entity_types or ENTITY_TYPES
    labels = ["O"]
    for t in types:
        labels.append(f"B-{t}")
        labels.append(f"I-{t}")
    return labels


DEFAULT_LABEL_LIST = build_label_list()


def span_to_bio(text: str, entities: List[dict]) -> List[str]:
    """
    Convert span annotations to character-level BIO tags.

    Args:
        text: Input text
        entities: List of entity dicts with start, end, label keys

    Returns:
        List of BIO tags, one per character
    """
    bio = ["O"] * len(text)
    for ent in sorted(entities, key=lambda e: e["start"]):
        s, e = ent["start"], ent["end"]
        if s < 0 or e > len(text) or s >= e:
            logger.warning(f"Invalid span [{s}:{e}] for text length {len(text)}, skipping")
            continue
        bio[s] = f"B-{ent['label']}"
        for i in range(s + 1, e):
            bio[i] = f"I-{ent['label']}"
    return bio


class NERDataset(Dataset):
    """
    Dataset for financial NER tasks.

    Loads JSON files with span-level entity annotations and provides
    character-level BIO label mappings for BERT token classification.
    """

    def __init__(self, file_path: Union[str, Path], label_list: Optional[List[str]] = None):
        self.file_path = Path(file_path)
        self.label_list = label_list or DEFAULT_LABEL_LIST
        self.label2id = {label: i for i, label in enumerate(self.label_list)}
        self.id2label = {i: label for i, label in enumerate(self.label_list)}
        self.data = self._load_and_validate()
        logger.info(f"Loaded {len(self.data)} NER samples from {file_path}")

    def _load_and_validate(self) -> List[dict]:
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        with open(self.file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for i, sample in enumerate(data):
            text = sample["text"]
            for ent in sample.get("entities", []):
                actual = text[ent["start"]:ent["end"]]
                expected = ent.get("text", actual)
                if actual != expected:
                    logger.warning(
                        f"Sample {i}: entity offset mismatch at [{ent['start']}:{ent['end']}]. "
                        f"Expected '{expected}', got '{actual}'"
                    )
        return data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        return self.data[idx]

    def get_label_info(self) -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
        return self.label_list, self.label2id, self.id2label


class NERDataCollator:
    """
    Data collator for NER that handles tokenization and BIO label alignment.

    For each sample:
    1. Split text into characters
    2. Assign BIO labels to characters from span annotations
    3. Tokenize with is_split_into_words=True (fast tokenizer)
       or regular tokenization with manual alignment (slow tokenizer)
    4. Align labels: first subword gets the char label, rest get -100
    5. [CLS] and [SEP] tokens get -100
    """

    def __init__(self, tokenizer, label2id: Dict[str, int], max_length: int = 512):
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length
        self._is_fast = getattr(tokenizer, "is_fast", False)

    def _align_fast(self, chars: List[str], char_label_ids: List[int]) -> Tuple[dict, List[int]]:
        """Align labels using fast tokenizer's word_ids()."""
        encoding = self.tokenizer(
            chars,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )
        word_ids = encoding.word_ids()
        aligned = []
        prev = None
        for wid in word_ids:
            if wid is None:
                aligned.append(-100)
            elif wid != prev:
                aligned.append(char_label_ids[wid] if wid < len(char_label_ids) else -100)
            else:
                aligned.append(-100)
            prev = wid
        return encoding, aligned

    def _align_slow(self, text: str, char_label_ids: List[int]) -> Tuple[dict, List[int]]:
        """Align labels using manual character-to-token mapping for slow tokenizer."""
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )
        tokens = self.tokenizer.convert_ids_to_tokens(encoding["input_ids"])
        aligned = []
        char_idx = 0
        for token in tokens:
            if token in (self.tokenizer.cls_token, self.tokenizer.sep_token,
                         self.tokenizer.pad_token):
                aligned.append(-100)
            elif token.startswith("##"):
                aligned.append(-100)
            else:
                if char_idx < len(char_label_ids):
                    aligned.append(char_label_ids[char_idx])
                    # Advance by token length (for multi-char tokens in extended vocab)
                    token_len = len(token.replace("##", ""))
                    char_idx += max(token_len, 1)
                else:
                    aligned.append(-100)
        return encoding, aligned

    def __call__(self, batch: List[dict]) -> Dict[str, torch.Tensor]:
        all_input_ids = []
        all_attention_masks = []
        all_labels = []

        for sample in batch:
            text = sample["text"]
            entities = sample.get("entities", [])
            chars = list(text)
            bio_tags = span_to_bio(text, entities)
            char_label_ids = [self.label2id.get(tag, 0) for tag in bio_tags]

            if self._is_fast:
                encoding, aligned = self._align_fast(chars, char_label_ids)
            else:
                encoding, aligned = self._align_slow(text, char_label_ids)

            all_input_ids.append(encoding["input_ids"])
            all_attention_masks.append(encoding["attention_mask"])
            all_labels.append(aligned)

        # Pad to max length in batch
        max_len = max(len(ids) for ids in all_input_ids)
        pad_id = self.tokenizer.pad_token_id or 0

        for i in range(len(all_input_ids)):
            pad_len = max_len - len(all_input_ids[i])
            all_input_ids[i] = all_input_ids[i] + [pad_id] * pad_len
            all_attention_masks[i] = all_attention_masks[i] + [0] * pad_len
            all_labels[i] = all_labels[i] + [-100] * pad_len

        return {
            "input_ids": torch.tensor(all_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(all_attention_masks, dtype=torch.long),
            "labels": torch.tensor(all_labels, dtype=torch.long),
        }
