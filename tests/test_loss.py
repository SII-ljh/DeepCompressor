"""Tests for loss.py."""

import torch

from deep_compressor.loss import DistillationLoss, compute_total_loss


def test_kl_zero_when_same():
    """KL divergence should be ~0 when student == teacher."""
    dl = DistillationLoss(temperature=2.0)
    logits = torch.randn(2, 10, 100)
    mask = torch.ones(2, 10)
    kl = dl.compute_kl_loss(logits, logits.clone(), mask)
    assert kl.item() < 1e-5


def test_kl_positive_when_different():
    dl = DistillationLoss(temperature=2.0)
    student = torch.randn(2, 10, 100)
    teacher = torch.randn(2, 10, 100)
    mask = torch.ones(2, 10)
    kl = dl.compute_kl_loss(student, teacher, mask)
    assert kl.item() > 0


def test_kl_respects_mask():
    dl = DistillationLoss(temperature=2.0)
    student = torch.randn(2, 10, 100)
    teacher = torch.randn(2, 10, 100)
    # Only last 5 tokens are answer tokens
    mask = torch.zeros(2, 10)
    mask[:, 5:] = 1.0
    kl_partial = dl.compute_kl_loss(student, teacher, mask)
    mask_full = torch.ones(2, 10)
    kl_full = dl.compute_kl_loss(student, teacher, mask_full)
    # They should differ since different positions are weighted
    assert not torch.isclose(kl_partial, kl_full)


def test_hidden_mse_ramp():
    dl = DistillationLoss(temperature=2.0, hidden_distill_layers=[0, 1],
                          hidden_distill_ramp_steps=100)
    student_h = [torch.randn(2, 10, 64) for _ in range(4)]
    teacher_h = [torch.randn(2, 10, 64) for _ in range(4)]
    mask = torch.ones(2, 10)

    loss_step0 = dl.compute_hidden_mse_loss(student_h, teacher_h, mask, global_step=0)
    loss_step50 = dl.compute_hidden_mse_loss(student_h, teacher_h, mask, global_step=50)
    loss_step100 = dl.compute_hidden_mse_loss(student_h, teacher_h, mask, global_step=100)

    assert loss_step0.item() < loss_step50.item()
    assert loss_step50.item() < loss_step100.item()


def test_total_loss_components():
    qa = torch.tensor(1.0)
    kl = torch.tensor(0.5)
    hidden = torch.tensor(0.3)
    result = compute_total_loss(qa, kl_loss=kl, hidden_mse_loss=hidden)
    assert "total" in result
    assert "qa_ce" in result
    assert "kl" in result
    assert "hidden_mse" in result
    assert result["total"].item() == pytest.approx(1.0 + 0.5 + 0.3)


def test_total_loss_without_optional():
    qa = torch.tensor(2.0)
    result = compute_total_loss(qa)
    assert result["total"].item() == pytest.approx(2.0)
    assert "kl" not in result


import pytest
