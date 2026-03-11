"""Datasets and collators for QA fine-tuning."""

import json
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


class QADataset(Dataset):
    """QA dataset for fine-tuning + distillation.

    Reads json with [{"context": ..., "question": ..., "answer": ...}, ...].
    """

    def __init__(self, data_path: str, tokenizer,
                 max_doc_tokens: int = 8192, max_question_tokens: int = 256,
                 max_answer_tokens: int = 512):
        self.tokenizer = tokenizer
        self.max_doc_tokens = max_doc_tokens
        self.max_question_tokens = max_question_tokens
        self.max_answer_tokens = max_answer_tokens

        with open(data_path) as f:
            self.data = json.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]

        doc = self.tokenizer(item["context"], truncation=True, max_length=self.max_doc_tokens,
                             return_tensors="pt", padding=False)
        question = self.tokenizer(item["question"], truncation=True, max_length=self.max_question_tokens,
                                  return_tensors="pt", padding=False)
        # Reserve 1 token for EOS so truncation leaves room
        answer = self.tokenizer(item["answer"], truncation=True, max_length=self.max_answer_tokens - 1,
                                return_tensors="pt", padding=False)
        answer_ids = answer["input_ids"].squeeze(0)

        # Append EOS token so the model learns to stop generating
        eos = torch.tensor([self.tokenizer.eos_token_id], dtype=answer_ids.dtype)
        answer_ids = torch.cat([answer_ids, eos])

        return {
            "doc_input_ids": doc["input_ids"].squeeze(0),
            "q_input_ids": question["input_ids"].squeeze(0),
            "answer_ids": answer_ids,
            "answer_labels": answer_ids.clone(),
            "question_text": item["question"],  # For eval sample display
            "answer_text": item["answer"],
        }


class PaddingCollator:
    """Pads variable-length fields to batch max length."""

    def __init__(self, pad_token_id: int = 0, label_pad_id: int = -100):
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id

    def __call__(self, batch: List[Dict]) -> Dict:
        result = {}
        keys = batch[0].keys()
        for key in keys:
            values = [sample[key] for sample in batch]
            # Non-tensor fields (e.g. answer_text) — collect as list
            if not isinstance(values[0], torch.Tensor):
                result[key] = values
                continue

            tensors = values
            max_len = max(t.shape[0] for t in tensors)

            is_label = "label" in key
            pad_val = self.label_pad_id if is_label else self.pad_token_id

            padded = []
            masks = []
            for t in tensors:
                pad_len = max_len - t.shape[0]
                padded_t = torch.cat([t, torch.full((pad_len,), pad_val, dtype=t.dtype)])
                mask = torch.cat([torch.ones(t.shape[0], dtype=torch.long),
                                  torch.zeros(pad_len, dtype=torch.long)])
                padded.append(padded_t)
                masks.append(mask)

            result[key] = torch.stack(padded)
            # Auto-generate attention mask for input_ids fields
            # doc_input_ids → doc_attention_mask, segment_ids → segment_attention_mask
            if "_ids" in key:
                mask_key = key.replace("_input_ids", "_attention_mask").replace(
                    "_ids", "_attention_mask")
                result[mask_key] = torch.stack(masks)

        return result
