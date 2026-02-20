"""Unit tests for eval.py metric functions and evaluation helpers."""

import torch
import torch.nn as nn

from deep_compressor.eval import (
    compute_exact_match,
    compute_f1,
    normalize_text,
)


# ── normalize_text ──────────────────────────────────────────────────────

class TestNormalizeText:
    def test_lowercase(self):
        assert normalize_text("Hello World") == "hello world"

    def test_strip_punctuation(self):
        assert normalize_text("$10M!") == "10m"

    def test_chinese_punctuation(self):
        assert normalize_text("你好，世界！") == "你好世界"

    def test_whitespace_collapse(self):
        assert normalize_text("  hello   world  ") == "hello world"

    def test_empty(self):
        assert normalize_text("") == ""

    def test_mixed_chinese_english(self):
        result = normalize_text("Revenue 收入: $10M")
        assert result == "revenue 收入 10m"


# ── compute_exact_match ──────────────────────────────────────────────────

class TestExactMatch:
    def test_exact(self):
        assert compute_exact_match("Hello", "hello") == 1.0

    def test_not_exact(self):
        assert compute_exact_match("Hello", "world") == 0.0

    def test_with_punctuation(self):
        assert compute_exact_match("$10M!", "10m") == 1.0

    def test_empty_both(self):
        assert compute_exact_match("", "") == 1.0

    def test_chinese(self):
        assert compute_exact_match("你好世界", "你好，世界！") == 1.0


# ── compute_f1 ──────────────────────────────────────────────────────────

class TestF1:
    def test_perfect_match(self):
        assert compute_f1("the cat sat", "the cat sat") == 1.0

    def test_partial_match(self):
        f1 = compute_f1("the cat", "the cat sat")
        # precision = 2/2 = 1.0, recall = 2/3, f1 = 2*(1*2/3)/(1+2/3) = 0.8
        assert abs(f1 - 0.8) < 1e-6

    def test_no_overlap(self):
        assert compute_f1("hello", "world") == 0.0

    def test_empty_both(self):
        assert compute_f1("", "") == 1.0

    def test_empty_pred(self):
        assert compute_f1("", "something") == 0.0

    def test_empty_gold(self):
        assert compute_f1("something", "") == 0.0

    def test_chinese_f1(self):
        # Character-level F1 for Chinese
        f1 = compute_f1("收入十亿", "收入二十亿")
        # pred chars: 收,入,十,亿  gold chars: 收,入,二,十,亿
        # common: 收,入,十,亿 (4)  precision=4/4=1  recall=4/5=0.8
        # f1 = 2*1*0.8/(1+0.8) = 8/9 ≈ 0.889
        assert abs(f1 - 8 / 9) < 1e-6


# ── evaluate_ntp shape test ─────────────────────────────────────────────

class _MockQwenOutput:
    def __init__(self, hidden_states, logits=None, loss=None):
        self.hidden_states = hidden_states
        self.logits = logits
        self.loss = loss


class _MockQwenModel(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_layers):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self._hidden_size = hidden_size
        self._num_layers = num_layers

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None,
                labels=None, output_hidden_states=False, use_cache=False, **kw):
        if inputs_embeds is not None:
            h = inputs_embeds
        else:
            h = self.model.embed_tokens(input_ids)
        all_hidden = [h] * (self._num_layers + 1) if output_hidden_states else None
        logits = self.lm_head(h)
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1), ignore_index=-100)
        return _MockQwenOutput(hidden_states=all_hidden, logits=logits, loss=loss)

    def generate(self, inputs_embeds=None, attention_mask=None,
                 max_new_tokens=16, do_sample=False, pad_token_id=0,
                 eos_token_id=None, **kw):
        B = inputs_embeds.shape[0]
        input_len = inputs_embeds.shape[1]
        # Fake: return input_len + max_new_tokens random ids
        return torch.randint(0, 100, (B, input_len + max_new_tokens))


def _make_tiny_model():
    from deep_compressor.config import (
        DeepCompressorConfig, FinBERTConfig, LossConfig,
        PerceiverConfig, ProjectionConfig, QwenConfig, TrainingConfig,
    )
    from deep_compressor.model import DeepCompressor

    cfg = DeepCompressorConfig(
        qwen=QwenConfig(model_name_or_path="tiny", hidden_size=64,
                        num_hidden_layers=4, vocab_size=100),
        finbert=FinBERTConfig(enabled=False, hidden_size=48),
        perceiver=PerceiverConfig(
            perceiver_dim=32, num_queries=8, num_heads=4, head_dim=8,
            stage_a_cross_layers=1, stage_a_self_layers=1,
            stage_b_layers=1, stage_c_cross_layers=1, stage_c_self_layers=1,
            ff_mult=2, dropout=0.0),
        projection=ProjectionConfig(down_hidden=48, up_hidden=48, dropout=0.0),
        loss=LossConfig(hidden_distill_layers=[1, 3]),
        training=TrainingConfig(stage=1, batch_size=2),
    )
    mock_qwen = _MockQwenModel(64, 100, 4)
    return DeepCompressor(cfg, qwen_model=mock_qwen), cfg


class _FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def batch_decode(self, ids, skip_special_tokens=True):
        return ["fake answer"] * ids.shape[0]


def test_evaluate_ntp_shape():
    """Verify evaluate_ntp returns perplexity and loss as floats."""
    from unittest.mock import MagicMock
    from deep_compressor.eval import evaluate_ntp
    from deep_compressor.data import PaddingCollator
    from torch.utils.data import DataLoader, TensorDataset

    model, cfg = _make_tiny_model()
    B, doc_len, seg_len = 2, 16, 12
    V = cfg.qwen.vocab_size

    # Build a tiny dataset (3 batches)
    samples = []
    for _ in range(4):
        samples.append({
            "doc_input_ids": torch.randint(0, V, (doc_len,)),
            "segment_ids": torch.randint(0, V, (seg_len,)),
            "segment_labels": torch.randint(0, V, (seg_len,)),
        })
    collator = PaddingCollator(pad_token_id=0)
    loader = DataLoader(samples, batch_size=2, collate_fn=collator)

    # Mock accelerator
    acc = MagicMock()
    acc.device = torch.device("cpu")
    acc.gather = lambda x: x  # no-op for single process

    metrics = evaluate_ntp(model, loader, acc)
    assert "perplexity" in metrics
    assert "loss" in metrics
    assert metrics["perplexity"] > 0
    assert metrics["loss"] > 0


def test_generate_answer_shape():
    """Verify generate_answer returns tensor with correct batch size."""
    model, cfg = _make_tiny_model()
    B = 2
    num_queries = cfg.perceiver.num_queries
    qwen_dim = cfg.qwen.hidden_size

    prefix_embeds = torch.randn(B, num_queries, qwen_dim)
    q_ids = torch.randint(0, cfg.qwen.vocab_size, (B, 8))
    q_mask = torch.ones(B, 8, dtype=torch.long)
    tokenizer = _FakeTokenizer()

    gen = model.generate_answer(prefix_embeds, q_ids, q_mask,
                                tokenizer=tokenizer, max_new_tokens=16)
    assert gen.shape[0] == B
    assert gen.shape[1] == 16  # max_new_tokens
