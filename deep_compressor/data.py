"""Datasets and collators for NTP pretraining and QA fine-tuning."""

import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset


class NTPDataset(Dataset):
    """Next-token prediction dataset for Stage 1 pretraining.

    Each sample: compress document → prefix → predict continuation segment.
    Reads jsonl with {"text": "..."} per line.

    Uses lazy loading: only byte offsets are stored in memory (~22 MB for 2.7M
    entries), and text is read from disk on demand.  This keeps RAM usage low
    even for multi-GB JSONL files.
    """

    def __init__(self, data_path: str, tokenizer, max_doc_tokens: int = 8192,
                 segment_len: int = 256, seed: int = 42, use_questions: bool = False):
        self.tokenizer = tokenizer
        self.max_doc_tokens = max_doc_tokens
        self.segment_len = segment_len
        self.rng = random.Random(seed)
        self.data_path = str(Path(data_path).resolve())
        self.use_questions = use_questions

        # Build byte-offset index (one pass, no f.tell() per line)
        self.offsets: List[int] = []
        pos = 0
        with open(self.data_path, "rb") as f:
            for line in f:
                if line.strip():
                    self.offsets.append(pos)
                pos += len(line)

        # Per-worker file handle (reopened after fork)
        self._fh = None
        self._fh_pid = -1

    def _read_line(self, idx: int) -> str:
        """Seek to offset and read one line.  Reopens the file if the PID
        changed (i.e. we are in a forked DataLoader worker)."""
        pid = os.getpid()
        if self._fh is None or self._fh_pid != pid:
            if self._fh is not None:
                try:
                    self._fh.close()
                except Exception:
                    pass
            self._fh = open(self.data_path, "rb")
            self._fh_pid = pid
        self._fh.seek(self.offsets[idx])
        return self._fh.readline().decode("utf-8")

    def __len__(self) -> int:
        return len(self.offsets)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = json.loads(self._read_line(idx))
        text = data["text"]
        tokens = self.tokenizer(text, truncation=True,
                                max_length=self.max_doc_tokens + self.segment_len,
                                return_tensors="pt", padding=False)
        input_ids = tokens["input_ids"].squeeze(0)
        total_len = input_ids.shape[0]

        # Split: doc part and continuation segment
        if total_len <= self.segment_len + 1:
            split_point = total_len // 2
        else:
            max_split = total_len - self.segment_len
            split_point = self.rng.randint(1, max(1, min(max_split, self.max_doc_tokens)))

        doc_ids = input_ids[:split_point]
        seg_ids = input_ids[split_point:]

        # Pass raw segment tokens as both input and labels.
        # HuggingFace ForCausalLMLoss handles the causal shift internally
        # (logits[i] predicts labels[i+1]).
        result = {
            "doc_input_ids": doc_ids,
            "segment_ids": seg_ids,
            "segment_labels": seg_ids.clone(),
        }

        # Add question if available and enabled
        if self.use_questions and "question" in data:
            q_tokens = self.tokenizer(
                data["question"],
                truncation=True,
                max_length=256,  # Max question length
                return_tensors="pt",
                padding=False
            )
            result["q_input_ids"] = q_tokens["input_ids"].squeeze(0)

        return result


class QADataset(Dataset):
    """QA dataset for Stage 2 fine-tuning + distillation.

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
