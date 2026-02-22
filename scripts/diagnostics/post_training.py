#!/usr/bin/env python
"""Post-training diagnostic experiments for Deep Compressor.

Exp 6 — Attention Pattern Analysis: Multi-batch entropy, Gini, Jaccard coverage.
Exp 7 — Compression Fidelity: Multi-batch cosine similarity + matched random baseline.
Exp 8 — Document Length Scaling: NTP loss vs document token length.
Exp 9 — Distillation Effectiveness: KL divergence, token prediction agreement, hidden MSE.

Requires a trained checkpoint and evaluation data.

Usage:
    python scripts/diagnostics/post_training.py \
        --config configs/benchmark.yaml \
        --checkpoint outputs/checkpoint-final/trainable_weights.pt \
        --eval_data data/qa_dev.json \
        --experiments 6,7,8,9
"""

from __future__ import annotations

import json
import os
import sys

# Ensure project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from scripts.diagnostics.common import (
    attention_entropy,
    base_parser,
    detect_device,
    finish_wandb,
    gini_coefficient,
    init_wandb,
    load_model,
    log_wandb,
    to_device,
)
from deep_compressor.config import DeepCompressorConfig
from deep_compressor.data import NTPDataset, PaddingCollator
from deep_compressor.loss import DistillationLoss


ALL_EXPERIMENTS = ["6", "7", "8", "9"]


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 6 — Attention Pattern Analysis (enhanced)
# ═══════════════════════════════════════════════════════════════════════

def run_attention_analysis(
    model,
    eval_loader: DataLoader,
    device: torch.device,
    max_batches: int = 10,
) -> dict:
    """Analyze attention patterns across multiple batches.

    Enhancements over original Exp 4:
      - Multi-batch aggregation
      - Gini coefficient for attention concentration
      - Jaccard coverage for query specialization
    """

    print("\n" + "=" * 76)
    print("  EXPERIMENT 6: Attention Pattern Analysis (enhanced)")
    print("=" * 76)

    model.eval()
    D = model.config.qwen.hidden_size

    # Accumulators
    all_entropies: dict[str, list[float]] = {}
    all_ginis: dict[str, list[float]] = {}
    all_jaccard_scores: list[float] = []

    num_batches = 0
    for batch in eval_loader:
        if num_batches >= max_batches:
            break
        batch = to_device(batch, device)
        num_batches += 1

        B = batch["doc_input_ids"].shape[0]
        doc_mask = batch["doc_attention_mask"]

        with torch.no_grad():
            byte_array = model.encode_document(
                batch["doc_input_ids"], batch["doc_attention_mask"]
            )

            # Use question encoding if available, else zero-pooled
            if "q_input_ids" in batch:
                queries = model.encode_question(
                    batch["q_input_ids"], batch["q_attention_mask"]
                )
            else:
                queries = model.query_init(
                    torch.zeros(B, D, device=device)
                )

            # Run through perceiver stages, collecting attention weights
            attn_weights = {}
            x = queries
            perceiver = model.perceiver

            if perceiver.enable_stage_a:
                for i, block in enumerate(perceiver.stage_a_cross):
                    if block.has_cross_attn:
                        ca = block.cross_attn
                        q_norm = ca.norm_q(x)
                        kv_norm = ca.norm_kv(byte_array)
                        H = ca.num_heads
                        q_proj = ca.to_q(q_norm).view(B, -1, H, ca.head_dim).transpose(1, 2)
                        k_proj = ca.to_k(kv_norm).view(B, -1, H, ca.head_dim).transpose(1, 2)
                        attn_logits = (q_proj @ k_proj.transpose(-2, -1)) * ca.scale
                        if doc_mask is not None:
                            attn_logits = attn_logits.masked_fill(
                                ~doc_mask[:, None, None, :].bool(), float("-inf"))
                        attn_probs = torch.softmax(attn_logits, dim=-1)
                        attn_weights[f"stage_a_cross_{i}"] = attn_probs
                    x = block(x, kv=byte_array, kv_mask=doc_mask)
                for block in perceiver.stage_a_self:
                    x = block(x)

            if perceiver.enable_stage_b:
                for block in perceiver.stage_b_self:
                    x = block(x)

            if perceiver.enable_stage_c:
                for i, block in enumerate(perceiver.stage_c_cross):
                    if block.has_cross_attn:
                        ca = block.cross_attn
                        q_norm = ca.norm_q(x)
                        kv_norm = ca.norm_kv(byte_array)
                        H = ca.num_heads
                        q_proj = ca.to_q(q_norm).view(B, -1, H, ca.head_dim).transpose(1, 2)
                        k_proj = ca.to_k(kv_norm).view(B, -1, H, ca.head_dim).transpose(1, 2)
                        attn_logits = (q_proj @ k_proj.transpose(-2, -1)) * ca.scale
                        if doc_mask is not None:
                            attn_logits = attn_logits.masked_fill(
                                ~doc_mask[:, None, None, :].bool(), float("-inf"))
                        attn_probs = torch.softmax(attn_logits, dim=-1)
                        attn_weights[f"stage_c_cross_{i}"] = attn_probs
                    x = block(x, kv=byte_array, kv_mask=doc_mask)

            # Compute metrics for each attention layer
            for name, attn in attn_weights.items():
                # Entropy: (B, H, Q)
                ent = attention_entropy(attn)
                mean_ent = ent.mean().item()
                all_entropies.setdefault(name, []).append(mean_ent)

                # Gini coefficient per head, averaged
                gini_per_head = []
                for h in range(attn.shape[1]):
                    for q in range(attn.shape[2]):
                        g = gini_coefficient(attn[0, h, q])
                        gini_per_head.append(g)
                mean_gini = sum(gini_per_head) / max(len(gini_per_head), 1)
                all_ginis.setdefault(name, []).append(mean_gini)

                # Jaccard overlap for cross-attention (query specialization)
                if "cross" in name:
                    # Top-K positions per query (K = 10% of seq len)
                    S = attn.shape[-1]
                    K = max(1, S // 10)
                    # Use first sample, average across heads
                    avg_attn = attn[0].mean(dim=0)  # (Q, S)
                    _, topk_indices = avg_attn.topk(K, dim=-1)  # (Q, K)
                    nq = topk_indices.shape[0]
                    jaccard_sum = 0.0
                    jaccard_count = 0
                    for qi in range(nq):
                        set_i = set(topk_indices[qi].tolist())
                        for qj in range(qi + 1, nq):
                            set_j = set(topk_indices[qj].tolist())
                            intersection = len(set_i & set_j)
                            union = len(set_i | set_j)
                            jaccard_sum += intersection / max(union, 1)
                            jaccard_count += 1
                    if jaccard_count > 0:
                        all_jaccard_scores.append(jaccard_sum / jaccard_count)

    # ── report ────────────────────────────────────────────────────────
    print(
        f"\n  Aggregated over {num_batches} batches:"
    )
    print(
        f"\n  {'Layer':<25}  {'Mean Entropy':>14}  {'Mean Gini':>12}"
    )
    print(f"  {'─' * 55}")

    results = {}
    for name in all_entropies:
        mean_ent = sum(all_entropies[name]) / len(all_entropies[name])
        mean_gini = sum(all_ginis[name]) / len(all_ginis[name])
        print(f"  {name:<25}  {mean_ent:>14.4f}  {mean_gini:>12.4f}")
        results[f"{name}_entropy"] = mean_ent
        results[f"{name}_gini"] = mean_gini

    if all_jaccard_scores:
        mean_jaccard = sum(all_jaccard_scores) / len(all_jaccard_scores)
        results["mean_jaccard_overlap"] = mean_jaccard
        print(f"\n  Mean Jaccard query overlap: {mean_jaccard:.4f}")
        if mean_jaccard < 0.3:
            print("  PASS — Queries attend to diverse document regions")
        elif mean_jaccard < 0.6:
            print("  INFO — Moderate query overlap")
        else:
            print("  WARNING — High query overlap — queries may be redundant")

    # Overall entropy check
    all_ent_vals = [v for vs in all_entropies.values() for v in vs]
    if all_ent_vals:
        avg_entropy = sum(all_ent_vals) / len(all_ent_vals)
        results["overall_avg_entropy"] = avg_entropy
        if avg_entropy < 0.5:
            print("  WARNING: Very low entropy — attention may be collapsed")
        elif avg_entropy > 5.0:
            print("  INFO: High entropy — attention broadly distributed")
        else:
            print("  PASS — Entropy in healthy range")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 7 — Compression Fidelity (enhanced)
# ═══════════════════════════════════════════════════════════════════════

def run_compression_fidelity(
    model,
    eval_loader: DataLoader,
    device: torch.device,
    max_batches: int = 10,
) -> dict:
    """Measure how well compressed prefixes preserve document information.

    Enhancements over original Exp 5:
      - Multi-batch aggregation
      - Matched-distribution random baseline (same mean/std as actual prefix)
    """

    print("\n" + "=" * 76)
    print("  EXPERIMENT 7: Compression Fidelity (enhanced)")
    print("=" * 76)

    model.eval()
    D = model.config.qwen.hidden_size

    cos_sims = []
    random_cos_sims = []
    matched_cos_sims = []
    max_query_sims = []
    prefix_norms = []
    doc_norms = []

    num_batches = 0
    for batch in eval_loader:
        if num_batches >= max_batches:
            break
        batch = to_device(batch, device)
        num_batches += 1

        doc_mask = batch["doc_attention_mask"]
        B = batch["doc_input_ids"].shape[0]

        with torch.no_grad():
            # Get document hidden states
            qwen_out = model.qwen(
                input_ids=batch["doc_input_ids"],
                attention_mask=batch["doc_attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
            doc_hidden = qwen_out.hidden_states[-1].detach()
            del qwen_out

            byte_array = model.down_proj(doc_hidden)

            if "q_input_ids" in batch:
                queries = model.encode_question(
                    batch["q_input_ids"], batch["q_attention_mask"]
                )
            else:
                queries = model.query_init(
                    torch.zeros(B, D, device=device)
                )

            latent = model.perceiver(queries, byte_array, byte_mask=doc_mask)
            prefix = model.up_mlp(latent)

            # Mean pooled representations
            prefix_pooled = prefix.mean(dim=1)
            mask_f = doc_mask.unsqueeze(-1).float()
            doc_pooled = (
                (doc_hidden * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
            )

            # Global cosine similarity
            cos_sim = F.cosine_similarity(prefix_pooled, doc_pooled, dim=-1)
            cos_sims.append(cos_sim.mean().item())

            # Random baseline (standard)
            random_prefix = torch.randn_like(prefix) * 0.02
            random_pooled = random_prefix.mean(dim=1)
            random_cos = F.cosine_similarity(random_pooled, doc_pooled, dim=-1)
            random_cos_sims.append(random_cos.mean().item())

            # Matched-distribution random baseline
            prefix_mean = prefix.mean()
            prefix_std = prefix.std()
            matched_random = torch.randn_like(prefix) * prefix_std + prefix_mean
            matched_pooled = matched_random.mean(dim=1)
            matched_cos = F.cosine_similarity(matched_pooled, doc_pooled, dim=-1)
            matched_cos_sims.append(matched_cos.mean().item())

            # Per-query max cosine similarity
            prefix_norm = F.normalize(prefix, dim=-1)
            doc_norm = F.normalize(doc_hidden, dim=-1)
            sim_matrix = torch.bmm(prefix_norm, doc_norm.transpose(1, 2))
            if doc_mask is not None:
                sim_matrix = sim_matrix.masked_fill(
                    ~doc_mask[:, None, :].bool(), float("-inf")
                )
            max_sim = sim_matrix.max(dim=-1).values.mean().item()
            max_query_sims.append(max_sim)

            prefix_norms.append(prefix.norm(dim=-1).mean().item())
            doc_norms.append(doc_hidden.norm(dim=-1).mean().item())

    # ── report ────────────────────────────────────────────────────────
    avg_cos = sum(cos_sims) / len(cos_sims)
    avg_random = sum(random_cos_sims) / len(random_cos_sims)
    avg_matched = sum(matched_cos_sims) / len(matched_cos_sims)
    avg_max_q = sum(max_query_sims) / len(max_query_sims)
    avg_prefix_norm = sum(prefix_norms) / len(prefix_norms)
    avg_doc_norm = sum(doc_norms) / len(doc_norms)

    improvement_random = avg_cos - avg_random
    improvement_matched = avg_cos - avg_matched

    print(f"\n  Aggregated over {num_batches} batches:")
    print(f"\n  {'Metric':<45}  {'Value':>12}")
    print(f"  {'─' * 60}")
    print(f"  {'Global cosine sim (prefix↔doc mean)':<45}  {avg_cos:>12.4f}")
    print(f"  {'Random baseline cosine sim':<45}  {avg_random:>12.4f}")
    print(f"  {'Matched-distribution random cosine sim':<45}  {avg_matched:>12.4f}")
    print(f"  {'Mean max per-query cosine sim':<45}  {avg_max_q:>12.4f}")
    print(f"  {'Prefix embedding norm (mean)':<45}  {avg_prefix_norm:>12.4f}")
    print(f"  {'Doc hidden states norm (mean)':<45}  {avg_doc_norm:>12.4f}")
    print(f"  {'─' * 60}")
    print(f"  {'Improvement over random':<45}  {improvement_random:>+12.4f}")
    print(f"  {'Improvement over matched random':<45}  {improvement_matched:>+12.4f}")

    print(f"\n  DIAGNOSIS:")
    if improvement_matched > 0.1:
        print("  PASS — Prefix captures meaningful structure beyond statistical matching")
    elif improvement_matched > 0:
        print("  MARGINAL — Some structured information, but weak")
    else:
        print("  FAIL — Prefix no better than matched random noise")

    return {
        "global_cos_sim": avg_cos,
        "random_cos_sim": avg_random,
        "matched_cos_sim": avg_matched,
        "mean_max_query_sim": avg_max_q,
        "improvement_over_random": improvement_random,
        "improvement_over_matched": improvement_matched,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 8 — Document Length Scaling
# ═══════════════════════════════════════════════════════════════════════

def run_length_scaling(
    model,
    data_path: str,
    config: DeepCompressorConfig,
    device: torch.device,
    length_buckets: list[int] | None = None,
    samples_per_bucket: int = 20,
    tokenizer=None,
) -> dict:
    """Measure NTP loss as a function of document length.

    Loads NTP data, buckets by token count, and runs forward pass per bucket.
    """

    print("\n" + "=" * 76)
    print("  EXPERIMENT 8: Document Length Scaling")
    print("=" * 76)

    model.eval()

    if length_buckets is None:
        length_buckets = [256, 512, 1024, 2048, 4096]

    if tokenizer is None:
        from transformers import AutoTokenizer as AT
        tokenizer = AT.from_pretrained(config.qwen.model_name_or_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    ds = NTPDataset(
        data_path,
        tokenizer,
        max_doc_tokens=max(length_buckets) + 256,
        segment_len=config.training.ntp_segment_len,
    )

    collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)

    # Bucket samples by document length
    print(f"\n  Scanning {len(ds)} samples for length distribution...")
    buckets: dict[int, list[int]] = {b: [] for b in length_buckets}

    for idx in range(len(ds)):
        sample = ds[idx]
        doc_len = sample["doc_input_ids"].shape[0]
        # Assign to the largest bucket that doesn't exceed doc_len
        for b in sorted(length_buckets):
            if doc_len >= b and len(buckets[b]) < samples_per_bucket:
                buckets[b].append(idx)
                break

    # ── run per bucket ────────────────────────────────────────────────
    results = {}
    print(
        f"\n  {'Bucket':<12}  {'Samples':>8}  {'Mean Loss':>12}  {'Std Loss':>12}"
    )
    print(f"  {'─' * 50}")

    bucket_losses = []
    bucket_lengths = []

    for bucket_len in sorted(length_buckets):
        indices = buckets[bucket_len]
        if len(indices) < 2:
            print(f"  {bucket_len:<12}  {len(indices):>8}  {'(skip)':>12}")
            continue

        subset = Subset(ds, indices)
        loader = DataLoader(
            subset, batch_size=min(4, len(indices)), shuffle=False, collate_fn=collator
        )

        batch_losses = []
        with torch.no_grad():
            for batch in loader:
                batch = to_device(batch, device)
                losses = model(
                    mode="ntp",
                    doc_input_ids=batch["doc_input_ids"],
                    doc_attention_mask=batch["doc_attention_mask"],
                    segment_ids=batch["segment_ids"],
                    segment_attention_mask=batch["segment_attention_mask"],
                    segment_labels=batch["segment_labels"],
                )
                batch_losses.append(losses["total"].item())

        mean_loss = sum(batch_losses) / len(batch_losses)
        std_loss = (
            (sum((l - mean_loss) ** 2 for l in batch_losses) / len(batch_losses))
            ** 0.5
            if len(batch_losses) > 1
            else 0.0
        )

        results[f"loss_{bucket_len}"] = mean_loss
        results[f"std_{bucket_len}"] = std_loss
        bucket_losses.append(mean_loss)
        bucket_lengths.append(bucket_len)

        print(f"  {bucket_len:<12}  {len(indices):>8}  {mean_loss:>12.4f}  {std_loss:>12.4f}")

    print(f"  {'─' * 50}")

    # ── degradation slope ─────────────────────────────────────────────
    if len(bucket_losses) >= 2:
        import math

        log_lengths = [math.log(l) for l in bucket_lengths]
        n = len(log_lengths)
        mean_x = sum(log_lengths) / n
        mean_y = sum(bucket_losses) / n
        cov_xy = sum(
            (x - mean_x) * (y - mean_y) for x, y in zip(log_lengths, bucket_losses)
        )
        var_x = sum((x - mean_x) ** 2 for x in log_lengths)
        slope = cov_xy / max(var_x, 1e-10)
        results["degradation_slope"] = slope

        print(f"\n  Degradation slope (loss vs log(length)): {slope:+.4f}")
        print(f"\n  DIAGNOSIS:")
        if slope < 0.05:
            print("  PASS — Loss stable across document lengths")
        elif slope < 0.2:
            print("  INFO — Moderate degradation with length")
        else:
            print("  WARNING — Significant loss increase with document length")
    else:
        print("\n  WARNING: Not enough buckets with data for slope analysis")

    return results


# ═══════════════════════════════════════════════════════════════════════
#  Experiment 9 — Distillation Effectiveness
# ═══════════════════════════════════════════════════════════════════════

def run_distillation_effectiveness(
    model,
    eval_loader: DataLoader,
    device: torch.device,
    max_batches: int = 10,
) -> dict:
    """Compare student and teacher outputs on evaluation data.

    Measures:
      - KL divergence between student and teacher logit distributions
      - Token prediction agreement (argmax match rate)
      - Hidden state MSE at distillation layers
    """

    print("\n" + "=" * 76)
    print("  EXPERIMENT 9: Distillation Effectiveness")
    print("=" * 76)

    model.eval()
    config = model.config
    D = config.qwen.hidden_size

    distill_loss = DistillationLoss(
        temperature=config.loss.kl_temperature,
        hidden_distill_layers=config.loss.hidden_distill_layers,
        hidden_distill_ramp_steps=1,  # full weight (post-training)
    )

    kl_divs = []
    agreement_rates = []
    hidden_mses = []

    num_batches = 0
    for batch in eval_loader:
        if num_batches >= max_batches:
            break

        # Skip if batch doesn't have QA fields
        if "q_input_ids" not in batch:
            continue

        batch = to_device(batch, device)
        num_batches += 1

        B = batch["doc_input_ids"].shape[0]

        with torch.no_grad():
            # ── Student path ──────────────────────────────────────────
            byte_array = model.encode_document(
                batch["doc_input_ids"], batch["doc_attention_mask"]
            )
            queries = model.encode_question(
                batch["q_input_ids"], batch["q_attention_mask"]
            )
            latent = model.perceiver(
                queries, byte_array, byte_mask=batch["doc_attention_mask"]
            )
            prefix = model.up_mlp(latent)

            suffix_ids = torch.cat(
                [batch["q_input_ids"], batch["answer_ids"]], dim=1
            )
            suffix_mask = torch.cat(
                [batch["q_attention_mask"], batch["answer_attention_mask"]], dim=1
            )

            q_len = batch["q_input_ids"].shape[1]
            q_labels = torch.full_like(batch["q_input_ids"], -100)
            full_labels = torch.cat([q_labels, batch["answer_labels"]], dim=1)

            student_out = model.decode(
                prefix, suffix_ids, suffix_mask,
                labels=full_labels, output_hidden_states=True,
            )
            prefix_len = prefix.shape[1]
            student_logits = student_out.logits[:, prefix_len:, :]

            # ── Teacher path (full doc + q + a through frozen Qwen) ───
            teacher_input_ids = torch.cat(
                [batch["doc_input_ids"], batch["q_input_ids"], batch["answer_ids"]],
                dim=1,
            )
            teacher_mask = torch.cat(
                [
                    batch["doc_attention_mask"],
                    batch["q_attention_mask"],
                    batch["answer_attention_mask"],
                ],
                dim=1,
            )

            teacher_out = model.qwen(
                input_ids=teacher_input_ids,
                attention_mask=teacher_mask,
                output_hidden_states=True,
                use_cache=False,
            )

            # Align: extract the q+a portion from teacher outputs
            doc_len = batch["doc_input_ids"].shape[1]
            teacher_logits = teacher_out.logits[:, doc_len:, :]

            # Trim to same length
            min_len = min(student_logits.shape[1], teacher_logits.shape[1])
            student_logits_trim = student_logits[:, :min_len, :]
            teacher_logits_trim = teacher_logits[:, :min_len, :]

            # Answer mask for q+a region
            answer_mask = torch.zeros(B, min_len, device=device)
            if min_len > q_len:
                answer_mask[:, q_len:] = batch["answer_attention_mask"][
                    :, : min_len - q_len
                ].float()

            # KL divergence
            kl = distill_loss.compute_kl_loss(
                student_logits_trim, teacher_logits_trim, answer_mask
            )
            kl_divs.append(kl.item())

            # Token prediction agreement (on answer tokens)
            student_preds = student_logits_trim.argmax(dim=-1)
            teacher_preds = teacher_logits_trim.argmax(dim=-1)
            match = (student_preds == teacher_preds).float() * answer_mask
            num_answer = answer_mask.sum().clamp(min=1)
            agreement = (match.sum() / num_answer).item()
            agreement_rates.append(agreement)

            # Hidden state MSE at distillation layers
            student_hidden = [
                h[:, prefix_len:, :][:, :min_len, :]
                for h in student_out.hidden_states
            ]
            teacher_hidden = [
                h[:, doc_len:, :][:, :min_len, :] for h in teacher_out.hidden_states
            ]
            shared_mask = torch.ones(B, min_len, device=device)
            if min_len > q_len:
                shared_mask[:, q_len:] = batch["answer_attention_mask"][
                    :, : min_len - q_len
                ].float()

            h_mse = distill_loss.compute_hidden_mse_loss(
                student_hidden, teacher_hidden, shared_mask, global_step=999999
            )
            hidden_mses.append(h_mse.item())

    if num_batches == 0:
        print("  WARNING: No QA batches found in eval data")
        return {}

    # ── report ────────────────────────────────────────────────────────
    avg_kl = sum(kl_divs) / len(kl_divs)
    avg_agree = sum(agreement_rates) / len(agreement_rates)
    avg_hmse = sum(hidden_mses) / len(hidden_mses)

    print(f"\n  Aggregated over {num_batches} batches:")
    print(f"\n  {'Metric':<45}  {'Value':>12}")
    print(f"  {'─' * 60}")
    print(f"  {'KL divergence (student ∥ teacher)':<45}  {avg_kl:>12.4f}")
    print(f"  {'Token prediction agreement':<45}  {avg_agree:>12.2%}")
    print(f"  {'Hidden state MSE (distill layers)':<45}  {avg_hmse:>12.4f}")
    print(f"  {'─' * 60}")

    print(f"\n  DIAGNOSIS:")
    if avg_agree > 0.8:
        print(
            f"  PASS — High teacher-student agreement ({avg_agree:.1%}); "
            f"distillation effective"
        )
    elif avg_agree > 0.5:
        print(
            f"  INFO — Moderate agreement ({avg_agree:.1%}); "
            f"room for improvement"
        )
    else:
        print(
            f"  WARNING — Low agreement ({avg_agree:.1%}); "
            f"distillation may need more training"
        )

    if avg_kl < 1.0:
        print(f"  KL divergence is low ({avg_kl:.4f}) — distributions are close")
    elif avg_kl < 5.0:
        print(f"  KL divergence is moderate ({avg_kl:.4f})")
    else:
        print(f"  KL divergence is high ({avg_kl:.4f}) — significant distribution gap")

    return {
        "kl_divergence": avg_kl,
        "token_agreement": avg_agree,
        "hidden_mse": avg_hmse,
    }


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    parser = base_parser("Deep Compressor post-training diagnostics (Exp 6-9)")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint (trainable_weights.pt or directory)",
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        required=True,
        help="Path to evaluation data (QA JSON for Exp 6/7/9, NTP JSONL for Exp 8)",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=0,
                        help="Limit eval samples (0 = all)")
    parser.add_argument("--max_batches", type=int, default=10,
                        help="Max batches per experiment")
    parser.add_argument(
        "--experiments",
        type=str,
        default="6,7,8,9",
        help="Comma-separated list of experiments to run (6-9)",
    )
    parser.add_argument(
        "--length_buckets",
        type=str,
        default="256,512,1024,2048,4096",
        help="Comma-separated document length buckets for Exp 8",
    )
    parser.add_argument(
        "--ntp_data",
        type=str,
        default=None,
        help="Path to NTP data for Exp 8 (defaults to --data_path)",
    )
    parser.set_defaults(wandb_project="dc-diagnostic-post")
    args = parser.parse_args()

    experiments = [e.strip() for e in args.experiments.split(",")]
    device = detect_device()

    wandb_run = init_wandb(
        enabled=args.wandb,
        project=args.wandb_project,
        run_name="post-training-diag",
        config={
            "checkpoint": args.checkpoint,
            "experiments": experiments,
        },
        entity=args.wandb_entity,
    )

    print(f"Device: {device}")
    config = DeepCompressorConfig.from_yaml(args.config)

    print("\nLoading model with checkpoint...")
    model = load_model(config, checkpoint_path=args.checkpoint, device=device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters()) - trainable
    print(f"  Trainable: {trainable:,}  |  Frozen: {frozen:,}")

    # ── prepare eval loader ───────────────────────────────────────────
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.qwen.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    eval_loader = None
    if any(e in experiments for e in ("6", "7", "9")):
        from scripts.diagnostics.common import prepare_qa_loader
        eval_loader, _ = prepare_qa_loader(
            config, args.eval_data, args.batch_size, max_samples=args.max_samples
        )
        print(f"\n  QA eval loader: {len(eval_loader)} batches")

    # ── experiments ───────────────────────────────────────────────────
    all_results = {}

    if "6" in experiments and eval_loader is not None:
        torch.manual_seed(config.training.seed)
        exp6 = run_attention_analysis(
            model, eval_loader, device, max_batches=args.max_batches
        )
        all_results["attention"] = exp6
        if wandb_run:
            log_wandb(
                wandb_run,
                {f"exp6/{k}": v for k, v in exp6.items() if isinstance(v, (int, float))},
            )

    if "7" in experiments and eval_loader is not None:
        torch.manual_seed(config.training.seed)
        exp7 = run_compression_fidelity(
            model, eval_loader, device, max_batches=args.max_batches
        )
        all_results["fidelity"] = exp7
        if wandb_run:
            log_wandb(wandb_run, {f"exp7/{k}": v for k, v in exp7.items()})

    if "8" in experiments:
        torch.manual_seed(config.training.seed)
        ntp_path = args.ntp_data or args.data_path
        length_buckets = [int(x) for x in args.length_buckets.split(",")]
        exp8 = run_length_scaling(
            model,
            ntp_path,
            config,
            device,
            length_buckets=length_buckets,
            tokenizer=tokenizer,
        )
        all_results["length_scaling"] = exp8
        if wandb_run:
            log_wandb(
                wandb_run,
                {f"exp8/{k}": v for k, v in exp8.items() if isinstance(v, (int, float))},
            )

    if "9" in experiments and eval_loader is not None:
        torch.manual_seed(config.training.seed)
        exp9 = run_distillation_effectiveness(
            model, eval_loader, device, max_batches=args.max_batches
        )
        all_results["distillation"] = exp9
        if wandb_run:
            log_wandb(wandb_run, {f"exp9/{k}": v for k, v in exp9.items()})

    finish_wandb(wandb_run)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
