"""Benchmark QA Dataset for external evaluation data.

Similar to QADataset but accepts in-memory records instead of a file path,
and returns multi-reference answers for metrics like compute_best_metric.
"""

from typing import Dict, List

import torch
from torch.utils.data import Dataset


class BenchmarkQADataset(Dataset):
    """QA dataset constructed from in-memory records.

    Each record should have: context, question, answer, answers (list).
    Compatible with PaddingCollator (non-tensor fields collected as lists).
    """

    def __init__(self, records: List[Dict], tokenizer,
                 max_doc_tokens: int = 8192, max_question_tokens: int = 256,
                 max_answer_tokens: int = 512):
        self.records = records
        self.tokenizer = tokenizer
        self.max_doc_tokens = max_doc_tokens
        self.max_question_tokens = max_question_tokens
        self.max_answer_tokens = max_answer_tokens

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict:
        item = self.records[idx]

        doc = self.tokenizer(
            item["context"], truncation=True,
            max_length=self.max_doc_tokens,
            return_tensors="pt", padding=False)
        question = self.tokenizer(
            item["question"], truncation=True,
            max_length=self.max_question_tokens,
            return_tensors="pt", padding=False)
        answer = self.tokenizer(
            item["answer"], truncation=True,
            max_length=self.max_answer_tokens - 1,
            return_tensors="pt", padding=False)
        answer_ids = answer["input_ids"].squeeze(0)

        eos = torch.tensor([self.tokenizer.eos_token_id], dtype=answer_ids.dtype)
        answer_ids = torch.cat([answer_ids, eos])

        # Multi-reference answers
        all_answers = item.get("answers", [item["answer"]])

        return {
            "doc_input_ids": doc["input_ids"].squeeze(0),
            "q_input_ids": question["input_ids"].squeeze(0),
            "answer_ids": answer_ids,
            "answer_labels": answer_ids.clone(),
            "answer_text": item["answer"],
            "all_answers": all_answers,
        }
