"""Tests for auto_batch.py."""

import os
import warnings
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from deep_compressor.auto_batch import (
    build_sample_batch_fn_lora,
    build_sample_batch_fn_qa,
    compute_gradient_accumulation,
    find_max_batch_size,
    sync_batch_size_across_ranks,
)


# ── compute_gradient_accumulation ─────────────────────────────────────


class TestComputeGradientAccumulation:
    def test_exact_divisible(self):
        # 4 * 8 = 32, 256 / 32 = 8 exactly
        assert compute_gradient_accumulation(4, 8, 256) == 8

    def test_non_divisible(self):
        # 3 * 8 = 24, ceil(256 / 24) = 11
        assert compute_gradient_accumulation(3, 8, 256) == 11

    def test_overshoot_returns_one(self):
        # 64 * 8 = 512 > 256, so just 1
        assert compute_gradient_accumulation(64, 8, 256) == 1

    def test_single_gpu(self):
        # 4 * 1 = 4, ceil(256 / 4) = 64
        assert compute_gradient_accumulation(4, 1, 256) == 64

    def test_min_one(self):
        # Even if per_step > target, result is at least 1
        assert compute_gradient_accumulation(512, 1, 256) == 1

    def test_small_target(self):
        # target=1, any batch/gpu combo should give 1
        assert compute_gradient_accumulation(4, 8, 1) == 1


# ── build_sample_batch_fn_qa ──────────────────────────────────────────


class TestBuildSampleBatchFnQA:
    def test_output_shapes(self, tiny_config):
        device = torch.device("cpu")
        build_fn = build_sample_batch_fn_qa(tiny_config, 1000, device)
        batch = build_fn(3)

        assert batch["doc_input_ids"].shape == (3, tiny_config.qwen.max_doc_tokens)
        assert batch["doc_attention_mask"].shape == (3, tiny_config.qwen.max_doc_tokens)
        assert batch["q_input_ids"].shape == (3, tiny_config.qwen.max_question_tokens)
        assert batch["q_attention_mask"].shape == (3, tiny_config.qwen.max_question_tokens)
        assert batch["answer_ids"].shape == (3, tiny_config.qwen.max_answer_tokens)
        assert batch["answer_attention_mask"].shape == (3, tiny_config.qwen.max_answer_tokens)
        assert batch["answer_labels"].shape == (3, tiny_config.qwen.max_answer_tokens)

    def test_values_in_range(self, tiny_config):
        device = torch.device("cpu")
        vocab = 500
        build_fn = build_sample_batch_fn_qa(tiny_config, vocab, device)
        batch = build_fn(2)

        assert batch["doc_input_ids"].max().item() < vocab
        assert batch["doc_input_ids"].min().item() >= 0

    def test_device_placement(self, tiny_config):
        device = torch.device("cpu")
        build_fn = build_sample_batch_fn_qa(tiny_config, 100, device)
        batch = build_fn(1)
        for v in batch.values():
            assert v.device == device


# ── build_sample_batch_fn_lora ────────────────────────────────────────


class TestBuildSampleBatchFnLora:
    def test_output_shapes(self):
        device = torch.device("cpu")
        build_fn = build_sample_batch_fn_lora(128, 1000, 0, device)
        batch = build_fn(4)

        assert batch["input_ids"].shape == (4, 128)
        assert batch["attention_mask"].shape == (4, 128)
        assert batch["labels"].shape == (4, 128)

    def test_attention_mask_all_ones(self):
        device = torch.device("cpu")
        build_fn = build_sample_batch_fn_lora(64, 500, 0, device)
        batch = build_fn(2)
        assert (batch["attention_mask"] == 1).all()


# ── find_max_batch_size ───────────────────────────────────────────────


class TestFindMaxBatchSize:
    def test_cpu_returns_default_with_warning(self):
        model = torch.nn.Linear(10, 10)
        build_fn = lambda bs: {"x": torch.randn(bs, 10)}
        fwd_fn = lambda m, b: m(b["x"]).sum()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = find_max_batch_size(
                model, build_fn, fwd_fn, torch.device("cpu"))
            assert result == 4
            assert len(w) == 1
            assert "not supported on cpu" in str(w[0].message).lower()

    def test_mps_returns_default_with_warning(self):
        model = torch.nn.Linear(10, 10)
        build_fn = lambda bs: {"x": torch.randn(bs, 10)}
        fwd_fn = lambda m, b: m(b["x"]).sum()

        fake_mps = SimpleNamespace(type="mps")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = find_max_batch_size(
                model, build_fn, fwd_fn, fake_mps)
            assert result == 4
            assert len(w) == 1

    def test_binary_search_finds_exact_max(self):
        """Binary search refines beyond power-of-2 to find true max."""
        model = torch.nn.Linear(10, 10)
        fake_device = SimpleNamespace(type="cuda")

        # OOM at >20, so true max is 20
        def build_fn(bs):
            return {"x": torch.randn(bs, 10)}

        def forward_fn(m, batch):
            if batch["x"].shape[0] > 20:
                raise RuntimeError(
                    "CUDA out of memory. Tried to allocate 2.00 GiB")
            return m(batch["x"]).sum()

        result = find_max_batch_size(
            model, build_fn, forward_fn, fake_device,
            max_batch=64, min_batch=1, mixed_precision="no",
        )
        # Phase 1: 64 OOM, 32 OOM, 16 fits -> lo=16, hi=32
        # Phase 2: binary search 16..32 -> finds 20
        assert result == 20

    def test_halving_on_oom(self):
        """Simulate OOM for large batches, binary search finds exact boundary."""
        model = torch.nn.Linear(10, 10)
        fake_device = SimpleNamespace(type="cuda")

        def build_fn(bs):
            return {"x": torch.randn(bs, 10)}

        def forward_fn(m, batch):
            if batch["x"].shape[0] > 16:
                raise RuntimeError(
                    "CUDA out of memory. Tried to allocate 2.00 GiB")
            return m(batch["x"]).sum()

        result = find_max_batch_size(
            model, build_fn, forward_fn, fake_device,
            max_batch=64, min_batch=1, mixed_precision="no",
        )
        assert result == 16

    def test_halving_to_minimum(self):
        """All sizes OOM except min_batch."""
        model = torch.nn.Linear(10, 10)
        fake_device = SimpleNamespace(type="cuda")

        def build_fn(bs):
            return {"x": torch.randn(bs, 10)}

        def forward_fn(m, batch):
            if batch["x"].shape[0] > 1:
                raise RuntimeError("CUDA out of memory.")
            return m(batch["x"]).sum()

        result = find_max_batch_size(
            model, build_fn, forward_fn, fake_device,
            max_batch=32, min_batch=1, mixed_precision="no",
        )
        assert result == 1

    def test_non_oom_error_reraises(self):
        """Non-OOM RuntimeErrors should propagate."""
        model = torch.nn.Linear(10, 10)
        fake_device = SimpleNamespace(type="cuda")

        def build_fn(bs):
            return {"x": torch.randn(bs, 10)}

        def forward_fn(m, batch):
            raise RuntimeError("Some unrelated error")

        with pytest.raises(RuntimeError, match="Some unrelated error"):
            find_max_batch_size(
                model, build_fn, forward_fn, fake_device,
                max_batch=8, min_batch=1, mixed_precision="no",
            )

    def test_max_batch_not_power_of_two(self):
        """max_batch=50: phase 1 finds 32, then tries 50 and succeeds."""
        model = torch.nn.Linear(10, 10)
        fake_device = SimpleNamespace(type="cuda")
        tried_sizes = []

        def build_fn(bs):
            tried_sizes.append(bs)
            return {"x": torch.randn(bs, 10)}

        def forward_fn(m, batch):
            return m(batch["x"]).sum()

        result = find_max_batch_size(
            model, build_fn, forward_fn, fake_device,
            max_batch=50, min_batch=1, mixed_precision="no",
        )
        # Phase 1 starts at 32, succeeds. Then tries max_batch=50, succeeds.
        assert tried_sizes[0] == 32
        assert result == 50

    def test_binary_search_respects_max_batch(self):
        """Binary search should not exceed max_batch even if memory allows."""
        model = torch.nn.Linear(10, 10)
        fake_device = SimpleNamespace(type="cuda")

        def build_fn(bs):
            return {"x": torch.randn(bs, 10)}

        # Everything fits — but max_batch caps at 40
        def forward_fn(m, batch):
            return m(batch["x"]).sum()

        result = find_max_batch_size(
            model, build_fn, forward_fn, fake_device,
            max_batch=40, min_batch=1, mixed_precision="no",
        )
        assert result == 40

    def test_binary_search_between_powers(self):
        """OOM at >24 with max_batch=64 should find exactly 24."""
        model = torch.nn.Linear(10, 10)
        fake_device = SimpleNamespace(type="cuda")

        def build_fn(bs):
            return {"x": torch.randn(bs, 10)}

        def forward_fn(m, batch):
            if batch["x"].shape[0] > 24:
                raise RuntimeError("CUDA out of memory.")
            return m(batch["x"]).sum()

        result = find_max_batch_size(
            model, build_fn, forward_fn, fake_device,
            max_batch=64, min_batch=1, mixed_precision="no",
        )
        # Phase 1: 64 OOM, 32 OOM, 16 fits -> lo=16, hi=32
        # Phase 2: 24 fits, 28 OOM, 26 OOM, 25 OOM -> lo=24
        assert result == 24


# ── sync_batch_size_across_ranks ─────────────────────────────────────


class TestSyncBatchSizeAcrossRanks:
    def test_single_gpu_passthrough(self):
        """WORLD_SIZE=1 (or unset) returns local value unchanged."""
        with mock.patch.dict(os.environ, {"WORLD_SIZE": "1"}):
            assert sync_batch_size_across_ranks(16) == 16

    def test_no_world_size_passthrough(self):
        """No WORLD_SIZE env var returns local value unchanged."""
        env = os.environ.copy()
        env.pop("WORLD_SIZE", None)
        with mock.patch.dict(os.environ, env, clear=True):
            assert sync_batch_size_across_ranks(32) == 32
