"""Unit tests for DeepCompressor model using a tiny mock Qwen."""

import torch
import torch.nn as nn

from deep_compressor.config import DeepCompressorConfig
from deep_compressor.model import DeepCompressor


class _MockQwenOutput:
    def __init__(self, hidden_states=None, last_hidden_state=None, logits=None, loss=None):
        self.hidden_states = hidden_states
        self.last_hidden_state = last_hidden_state
        self.logits = logits
        self.loss = loss


class _MockQwenBaseModel(nn.Module):
    """Mock base model (self.qwen.model) — returns hidden states without LM head."""

    def __init__(self, embed_tokens, hidden_size, num_layers):
        super().__init__()
        self.embed_tokens = embed_tokens
        self._hidden_size = hidden_size
        self._num_layers = num_layers

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None,
                output_hidden_states=False, use_cache=False, **kwargs):
        if inputs_embeds is not None:
            h = inputs_embeds
        else:
            h = self.embed_tokens(input_ids)
        all_hidden = [h] * (self._num_layers + 1) if output_hidden_states else None
        return _MockQwenOutput(hidden_states=all_hidden, last_hidden_state=h)


class _MockQwenModel(nn.Module):
    """Minimal mock that behaves like Qwen2ForCausalLM for testing."""

    def __init__(self, hidden_size: int, vocab_size: int, num_layers: int):
        super().__init__()
        embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.model = _MockQwenBaseModel(embed_tokens, hidden_size, num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self._hidden_size = hidden_size
        self._num_layers = num_layers

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None,
                labels=None, output_hidden_states=False, use_cache=False, **kwargs):
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
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return _MockQwenOutput(hidden_states=all_hidden, last_hidden_state=h,
                               logits=logits, loss=loss)


def _make_model(config: DeepCompressorConfig) -> DeepCompressor:
    mock_qwen = _MockQwenModel(config.qwen.hidden_size, config.qwen.vocab_size,
                                config.qwen.num_hidden_layers)
    return DeepCompressor(config, qwen_model=mock_qwen)


def test_encode_decode_shapes(tiny_config, batch_size, doc_len, q_len):
    cfg = tiny_config
    model = _make_model(cfg)

    doc_ids = torch.randint(0, cfg.qwen.vocab_size, (batch_size, doc_len))
    doc_mask = torch.ones(batch_size, doc_len, dtype=torch.long)

    byte_array = model.encode_document(doc_ids, doc_mask)
    assert byte_array.shape == (batch_size, doc_len, cfg.perceiver.perceiver_dim)

    q_ids = torch.randint(0, cfg.qwen.vocab_size, (batch_size, q_len))
    q_mask = torch.ones(batch_size, q_len, dtype=torch.long)
    queries = model.encode_question(q_ids, q_mask)
    assert queries.shape == (batch_size, cfg.perceiver.num_queries, cfg.perceiver.perceiver_dim)

    latent = model.compress(queries, byte_array, byte_mask=doc_mask)
    assert latent.shape == (batch_size, cfg.perceiver.num_queries, cfg.perceiver.perceiver_dim)

    prefix = model.up_mlp(latent)
    assert prefix.shape == (batch_size, cfg.perceiver.num_queries, cfg.qwen.hidden_size)


def test_frozen_params(tiny_config):
    model = _make_model(tiny_config)
    for name, p in model.qwen.named_parameters():
        assert not p.requires_grad, f"Qwen param {name} should be frozen"


def test_trainable_params(tiny_config):
    model = _make_model(tiny_config)
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    # down_proj and up_mlp are identity by default (no params) — only check core trainable modules
    assert any("query_init" in n for n in trainable)
    assert any("perceiver" in n for n in trainable)


def test_forward_qa_returns_finite_loss(tiny_config, batch_size, doc_len, q_len):
    cfg = tiny_config
    model = _make_model(cfg)
    a_len = 8

    losses = model.forward_qa(
        doc_input_ids=torch.randint(0, cfg.qwen.vocab_size, (batch_size, doc_len)),
        doc_attention_mask=torch.ones(batch_size, doc_len, dtype=torch.long),
        q_input_ids=torch.randint(0, cfg.qwen.vocab_size, (batch_size, q_len)),
        q_attention_mask=torch.ones(batch_size, q_len, dtype=torch.long),
        answer_ids=torch.randint(0, cfg.qwen.vocab_size, (batch_size, a_len)),
        answer_attention_mask=torch.ones(batch_size, a_len, dtype=torch.long),
        answer_labels=torch.randint(0, cfg.qwen.vocab_size, (batch_size, a_len)),
    )
    assert torch.isfinite(losses["total"])
    assert "qa_ce" in losses


def test_forward_qa_backward(tiny_config, batch_size, doc_len, q_len):
    cfg = tiny_config
    model = _make_model(cfg)
    a_len = 8

    losses = model.forward_qa(
        doc_input_ids=torch.randint(0, cfg.qwen.vocab_size, (batch_size, doc_len)),
        doc_attention_mask=torch.ones(batch_size, doc_len, dtype=torch.long),
        q_input_ids=torch.randint(0, cfg.qwen.vocab_size, (batch_size, q_len)),
        q_attention_mask=torch.ones(batch_size, q_len, dtype=torch.long),
        answer_ids=torch.randint(0, cfg.qwen.vocab_size, (batch_size, a_len)),
        answer_attention_mask=torch.ones(batch_size, a_len, dtype=torch.long),
        answer_labels=torch.randint(0, cfg.qwen.vocab_size, (batch_size, a_len)),
    )
    losses["total"].backward()

    # Check that trainable params have gradients
    has_grad = False
    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is not None:
            has_grad = True
            break
    assert has_grad, "No trainable parameter received gradients"
