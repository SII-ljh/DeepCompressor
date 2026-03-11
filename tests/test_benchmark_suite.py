"""Tests for the benchmark suite: ROUGE-L, multi-ref metrics,
BenchmarkQADataset, build_model_for_ratio, and LongBench adapter."""

import json
import torch
import torch.nn as nn
import pytest
from unittest.mock import patch, MagicMock

from deep_compressor.eval import (
    compute_rouge_l,
    compute_best_metric,
    compute_exact_match,
    compute_f1,
    LatencyTimer,
)
from deep_compressor.benchmarks.dataset import BenchmarkQADataset
from deep_compressor.benchmarks.longbench import (
    LONGBENCH_QA_SUBSETS,
    _parse_answers,
    load_from_local,
    save_as_internal_format,
)


# ── ROUGE-L ──────────────────────────────────────────────────────────

class TestRougeL:
    def test_perfect_match(self):
        assert compute_rouge_l("the cat sat", "the cat sat") == 1.0

    def test_partial_match(self):
        # pred: [the, cat], gold: [the, cat, sat]
        # LCS = [the, cat] = 2
        # precision = 2/2 = 1.0, recall = 2/3
        # F1 = 2 * 1 * (2/3) / (1 + 2/3) = 0.8
        rouge = compute_rouge_l("the cat", "the cat sat")
        assert abs(rouge - 0.8) < 1e-6

    def test_no_overlap(self):
        assert compute_rouge_l("hello", "world") == 0.0

    def test_empty_both(self):
        assert compute_rouge_l("", "") == 1.0

    def test_empty_pred(self):
        assert compute_rouge_l("", "something") == 0.0

    def test_empty_gold(self):
        assert compute_rouge_l("something", "") == 0.0

    def test_chinese(self):
        # pred: 收,入,十,亿  gold: 收,入,二,十,亿
        # LCS = 收,入,十,亿 = 4
        # precision = 4/4 = 1.0, recall = 4/5 = 0.8
        # F1 = 2 * 1 * 0.8 / 1.8 = 8/9
        rouge = compute_rouge_l("收入十亿", "收入二十亿")
        assert abs(rouge - 8 / 9) < 1e-6

    def test_subsequence_order_matters(self):
        # pred: [a, b, c], gold: [c, b, a]
        # LCS = 1 (only one of a/b/c in order)
        # precision = 1/3, recall = 1/3
        # F1 = 2*(1/3)*(1/3)/(2/3) = 1/3
        rouge = compute_rouge_l("a b c", "c b a")
        assert abs(rouge - 1 / 3) < 1e-6


# ── compute_best_metric ──────────────────────────────────────────────

class TestBestMetric:
    def test_single_ref(self):
        score = compute_best_metric("hello", ["hello"], compute_exact_match)
        assert score == 1.0

    def test_multi_ref_takes_max(self):
        score = compute_best_metric("cat", ["dog", "cat"], compute_exact_match)
        assert score == 1.0

    def test_multi_ref_no_match(self):
        score = compute_best_metric("cat", ["dog", "fish"], compute_exact_match)
        assert score == 0.0

    def test_empty_golds(self):
        score = compute_best_metric("anything", [], compute_exact_match)
        assert score == 0.0

    def test_f1_multi_ref(self):
        score = compute_best_metric("the cat sat",
                                    ["the dog ran", "the cat sat on mat"],
                                    compute_f1)
        # Best match is "the cat sat on mat" -> F1 should be > match with "the dog ran"
        assert score > 0.5

    def test_rouge_multi_ref(self):
        score = compute_best_metric("the cat sat",
                                    ["abc", "the cat sat"],
                                    compute_rouge_l)
        assert score == 1.0


# ── LatencyTimer ─────────────────────────────────────────────────────

class TestLatencyTimer:
    def test_cpu_timing(self):
        with LatencyTimer(device=torch.device("cpu")) as timer:
            # Do some trivial work
            _ = sum(range(1000))
        assert timer.elapsed_ms >= 0

    def test_none_device(self):
        with LatencyTimer(device=None) as timer:
            _ = 1 + 1
        assert timer.elapsed_ms >= 0


# ── BenchmarkQADataset ───────────────────────────────────────────────

class _SimpleTokenizer:
    """Minimal tokenizer for testing."""
    pad_token_id = 0
    eos_token_id = 2

    def __call__(self, text, truncation=True, max_length=512,
                 return_tensors="pt", padding=False):
        # Produce deterministic token ids from text length
        n = min(len(text.split()), max_length)
        n = max(n, 1)
        ids = torch.arange(1, n + 1)
        return {"input_ids": ids.unsqueeze(0)}

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["decoded"] * ids.shape[0]


class TestBenchmarkQADataset:
    def test_from_records(self):
        records = [
            {"context": "some long document text here",
             "question": "what is this",
             "answer": "a test",
             "answers": ["a test", "another answer"]},
            {"context": "another document",
             "question": "what",
             "answer": "that",
             "answers": ["that"]},
        ]
        tokenizer = _SimpleTokenizer()
        ds = BenchmarkQADataset(records, tokenizer, max_doc_tokens=128,
                                max_question_tokens=32, max_answer_tokens=64)
        assert len(ds) == 2

        item = ds[0]
        assert "doc_input_ids" in item
        assert "q_input_ids" in item
        assert "answer_ids" in item
        assert "answer_labels" in item
        assert "answer_text" in item
        assert "all_answers" in item
        assert item["all_answers"] == ["a test", "another answer"]
        assert item["answer_text"] == "a test"
        # answer_ids should end with eos token
        assert item["answer_ids"][-1].item() == tokenizer.eos_token_id

    def test_missing_answers_field(self):
        records = [
            {"context": "doc", "question": "q", "answer": "a"},
        ]
        tokenizer = _SimpleTokenizer()
        ds = BenchmarkQADataset(records, tokenizer)
        item = ds[0]
        assert item["all_answers"] == ["a"]


# ── build_model_for_ratio ────────────────────────────────────────────

class _MockQwenOutput:
    def __init__(self, hidden_states=None, last_hidden_state=None,
                 logits=None, loss=None):
        self.hidden_states = hidden_states
        self.last_hidden_state = last_hidden_state
        self.logits = logits
        self.loss = loss


class _MockQwenBaseModel(nn.Module):
    def __init__(self, embed_tokens, hidden_size, num_layers):
        super().__init__()
        self.embed_tokens = embed_tokens
        self._hidden_size = hidden_size
        self._num_layers = num_layers

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None,
                output_hidden_states=False, use_cache=False, **kw):
        if inputs_embeds is not None:
            h = inputs_embeds
        else:
            h = self.embed_tokens(input_ids)
        all_hidden = ([h] * (self._num_layers + 1)
                      if output_hidden_states else None)
        return _MockQwenOutput(hidden_states=all_hidden, last_hidden_state=h)


class _MockQwenModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_layers):
        super().__init__()
        embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model = _MockQwenBaseModel(embed_tokens, hidden_size, num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self._hidden_size = hidden_size
        self._num_layers = num_layers

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None,
                labels=None, output_hidden_states=False, use_cache=False, **kw):
        if inputs_embeds is not None:
            h = inputs_embeds
        else:
            h = self.model.embed_tokens(input_ids)
        all_hidden = ([h] * (self._num_layers + 1)
                      if output_hidden_states else None)
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1), ignore_index=-100)
        return _MockQwenOutput(hidden_states=all_hidden,
                               last_hidden_state=h, logits=logits, loss=loss)

    def generate(self, inputs_embeds=None, attention_mask=None,
                 max_new_tokens=16, do_sample=False, pad_token_id=0,
                 eos_token_id=None, **kw):
        B = inputs_embeds.shape[0]
        return torch.randint(0, 100, (B, max_new_tokens))


def _make_config_and_qwen(num_queries=8):
    from deep_compressor.config import (
        DeepCompressorConfig, LossConfig,
        PerceiverConfig, ProjectionConfig, QwenConfig, TrainingConfig,
    )
    cfg = DeepCompressorConfig(
        qwen=QwenConfig(model_name_or_path="tiny", hidden_size=64,
                        num_hidden_layers=4, vocab_size=100),
        perceiver=PerceiverConfig(
            perceiver_dim=64, num_queries=num_queries, num_heads=4,
            head_dim=16, stage_a_cross_layers=1, stage_a_self_layers=1,
            stage_b_layers=1, stage_c_cross_layers=1, stage_c_self_layers=1,
            ff_mult=2, dropout=0.0),
        projection=ProjectionConfig(down_hidden=48, up_hidden=48, dropout=0.0),
        loss=LossConfig(hidden_distill_layers=[1, 3]),
        training=TrainingConfig(batch_size=2),
    )
    qwen = _MockQwenModel(64, 100, 4)
    return cfg, qwen


class TestBuildModelForRatio:
    def test_same_queries(self):
        """Loading with same num_queries should work exactly."""
        from deep_compressor.model import DeepCompressor
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                        "scripts"))
        from benchmark_suite import build_model_for_ratio

        cfg, qwen = _make_config_and_qwen(num_queries=8)
        model = DeepCompressor(cfg, qwen_model=qwen)

        # Save checkpoint
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
            trainable = {k: v for k, v in model.state_dict().items()
                         if not k.startswith("qwen.")}
            torch.save(trainable, ckpt_path)

        try:
            new_model = build_model_for_ratio(cfg, qwen, ckpt_path, 8,
                                              torch.device("cpu"))
            assert new_model.query_init.base_queries.shape[0] == 8
        finally:
            os.unlink(ckpt_path)

    def test_fewer_queries(self):
        """Loading with fewer queries should slice."""
        from deep_compressor.model import DeepCompressor
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                        "scripts"))
        from benchmark_suite import build_model_for_ratio

        cfg, qwen = _make_config_and_qwen(num_queries=8)
        model = DeepCompressor(cfg, qwen_model=qwen)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
            trainable = {k: v for k, v in model.state_dict().items()
                         if not k.startswith("qwen.")}
            torch.save(trainable, ckpt_path)

        try:
            new_model = build_model_for_ratio(cfg, qwen, ckpt_path, 4,
                                              torch.device("cpu"))
            assert new_model.query_init.base_queries.shape[0] == 4
        finally:
            os.unlink(ckpt_path)

    def test_more_queries(self):
        """Loading with more queries should tile."""
        from deep_compressor.model import DeepCompressor
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..",
                                        "scripts"))
        from benchmark_suite import build_model_for_ratio

        cfg, qwen = _make_config_and_qwen(num_queries=8)
        model = DeepCompressor(cfg, qwen_model=qwen)

        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            ckpt_path = f.name
            trainable = {k: v for k, v in model.state_dict().items()
                         if not k.startswith("qwen.")}
            torch.save(trainable, ckpt_path)

        try:
            new_model = build_model_for_ratio(cfg, qwen, ckpt_path, 16,
                                              torch.device("cpu"))
            assert new_model.query_init.base_queries.shape[0] == 16
        finally:
            os.unlink(ckpt_path)


# ── LongBench adapter ───────────────────────────────────────────────

class TestLongBenchAdapter:
    def test_subsets_defined(self):
        assert "hotpotqa" in LONGBENCH_QA_SUBSETS
        assert "multifieldqa_zh" in LONGBENCH_QA_SUBSETS
        assert "multifieldqa_en" in LONGBENCH_QA_SUBSETS
        assert "narrativeqa" in LONGBENCH_QA_SUBSETS

    def test_parse_answers_list(self):
        assert _parse_answers(["a", "b"]) == ["a", "b"]

    def test_parse_answers_json_string(self):
        assert _parse_answers('["a", "b"]') == ["a", "b"]

    def test_parse_answers_plain_string(self):
        assert _parse_answers("just text") == ["just text"]

    def test_save_and_load_local(self, tmp_path):
        records = [
            {"context": "doc", "question": "q", "answer": "a",
             "answers": ["a", "b"], "source": "test"},
        ]
        path = str(tmp_path / "test.json")
        save_as_internal_format(records, path)
        loaded = load_from_local(path)
        assert len(loaded) == 1
        assert loaded[0]["answers"] == ["a", "b"]

    def test_load_longbench_mock(self):
        """Test load_longbench_subset with mocked HuggingFace datasets."""
        import sys
        from deep_compressor.benchmarks.longbench import load_longbench_subset

        mock_data = [
            {"context": "Long document text...",
             "input": "What is the main topic?",
             "answers": '["topic A", "topic B"]'},
            {"context": "Another document.",
             "input": "Summarize.",
             "answers": '["summary"]'},
        ]

        mock_datasets = MagicMock()
        mock_datasets.load_dataset.return_value = mock_data
        with patch.dict(sys.modules, {"datasets": mock_datasets}):
            records = load_longbench_subset("hotpotqa")

        assert len(records) == 2
        assert records[0]["context"] == "Long document text..."
        assert records[0]["question"] == "What is the main topic?"
        assert records[0]["answers"] == ["topic A", "topic B"]
        assert records[0]["answer"] == "topic A"
        assert records[0]["source"] == "hotpotqa"
