#!/usr/bin/env python3
"""Overfit test: can DeepCompressor memorize 10 financial QA pairs with exact numbers?

This is the fastest feasibility check for the architecture. If the model can't
even overfit 10 samples (especially numerical answers), the compression approach
is fundamentally broken for financial QA.

Pipeline:
  1. Create 10 hand-crafted financial QA pairs (Chinese, with precise numbers)
  2. Write them as NTP jsonl + QA json to a temp directory
  3. Stage 1: NTP overfit on the 10 documents (~200 steps) — make prefix meaningful
  4. Stage 2: QA overfit on the 10 QA pairs (~500 steps) — learn to extract numbers
     - Periodically evaluate all 10 samples to track answer evolution
  5. Print report showing how answers change during training

Usage:
  python scripts/overfit_test.py
  python scripts/overfit_test.py --ntp_steps 300 --qa_steps 800
  python scripts/overfit_test.py --eval_every 100   # evaluate every 100 QA steps
  python scripts/overfit_test.py --skip_ntp  # skip Stage 1 if you have a checkpoint
"""

import argparse
import json
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from deep_compressor.config import (
    AblationConfig,
    DeepCompressorConfig,
    FinBERTConfig,
    LossConfig,
    PerceiverConfig,
    ProjectionConfig,
    QwenConfig,
    TrainingConfig,
)
from deep_compressor.eval import compute_exact_match, compute_f1, normalize_text
from deep_compressor.model import DeepCompressor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("overfit_test")

# ═══════════════════════════════════════════════════════════════════════
# 12 hand-crafted financial QA pairs — mixed difficulty levels:
#   - Simple numeric extraction (baseline)
#   - Sentence-length answers (numbers + text combined)
#   - Logical reasoning / comparison / inference
# ═══════════════════════════════════════════════════════════════════════
FINANCIAL_QA = [
    # ── Type A: Simple numeric extraction (3 samples) ──
    {
        "context": (
            "贵州茅台2023年年报显示，公司全年实现营业收入1505.6亿元，同比增长18.04%。"
            "归属于上市公司股东的净利润为747.34亿元，同比增长19.16%。"
            "其中茅台酒实现收入1346.24亿元，系列酒实现收入148.09亿元。"
            "公司全年销售毛利率为92.06%，较上年同期提升0.76个百分点。"
        ),
        "question": "贵州茅台2023年净利润是多少？",
        "answer": "747.34亿元",
    },
    {
        "context": (
            "招商银行2023年实现营业收入3391.23亿元，同比下降1.64%。"
            "净利润1466.02亿元，同比增长6.22%。不良贷款率0.95%，拨备覆盖率437.70%。"
            "零售金融业务收入1903.39亿元，占总营收的56.13%。"
            "管理零售客户总资产达13.32万亿元，较上年末增长8.93%。"
        ),
        "question": "招商银行2023年的拨备覆盖率是多少？",
        "answer": "437.70%",
    },
    {
        "context": (
            "美联储在2023年12月议息会议上维持联邦基金利率在5.25%-5.50%区间不变。"
            "点阵图显示2024年可能降息3次，合计75个基点。"
            "会议预计2024年美国GDP增速为1.4%，核心PCE通胀为2.4%。"
            "失业率预测为4.1%。美联储资产负债表规模约为7.7万亿美元。"
        ),
        "question": "美联储2023年12月的联邦基金利率区间是多少？",
        "answer": "5.25%-5.50%",
    },
    # ── Type B: Sentence-length answers (numbers + text) (5 samples) ──
    {
        "context": (
            "比亚迪2023年全年汽车销量达302.44万辆，同比增长61.9%。"
            "其中纯电动车型销量为157.48万辆，插电式混合动力车型销量为143.81万辆。"
            "公司全年营业收入6023.15亿元，净利润300.41亿元，同比增长80.72%。"
            "研发投入达到395.75亿元，占营收比例为6.57%。"
        ),
        "question": "比亚迪2023年的销量构成是怎样的？",
        "answer": "纯电动车型157.48万辆，插电式混合动力车型143.81万辆，合计302.44万辆",
    },
    {
        "context": (
            "宁德时代发布2023年三季报，前三季度实现营业收入2946.77亿元，"
            "同比下降6.35%。净利润为311.45亿元，同比增长77.05%。"
            "第三季度单季营收1054.31亿元，净利润为104.28亿元。"
            "公司动力电池系统销量为203.8GWh，储能电池系统销量为51.5GWh。"
        ),
        "question": "宁德时代2023年前三季度呈现了怎样的经营特征？",
        "answer": "营收同比下降6.35%但净利润同比增长77.05%，呈现增利不增收的特征",
    },
    {
        "context": (
            "腾讯控股2023年全年实现营收6090.15亿元，同比增长10%。"
            "非国际财务报告准则下的净利润为1576.88亿元，同比增长36%。"
            "微信及WeChat的合并月活跃账户数达到13.43亿。"
            "金融科技及企业服务业务收入2037.64亿元，占总收入的33.46%。"
        ),
        "question": "腾讯的金融科技及企业服务业务表现如何？",
        "answer": "该业务实现收入2037.64亿元，占总收入的33.46%，是第二大收入来源",
    },
    {
        "context": (
            "隆基绿能2023年全年实现营业收入1268.63亿元，同比增长8.55%。"
            "归属于上市公司股东的净利润为107.51亿元，同比下降27.41%。"
            "单晶硅片出货量为106.37GW，单晶电池出货量为54.82GW，"
            "单晶组件出货量为67.52GW。公司研发投入为77.21亿元。"
        ),
        "question": "隆基绿能2023年的盈利状况如何？",
        "answer": "营收增长8.55%至1268.63亿元，但净利润下降27.41%至107.51亿元，增收不增利",
    },
    {
        "context": (
            "中国人民银行决定于2024年2月5日下调金融机构存款准备金率0.5个百分点。"
            "此次降准后，金融机构加权平均存款准备金率约为7.0%。"
            "本次降准将释放长期流动性约1万亿元。"
            "央行同时宣布1月25日起下调支农再贷款、支小再贷款利率0.25个百分点。"
        ),
        "question": "央行2024年2月采取了哪些宽松措施？",
        "answer": "降准0.5个百分点释放约1万亿元流动性，同时下调支农支小再贷款利率0.25个百分点",
    },
    # ── Type C: Logical reasoning / comparison / inference (4 samples) ──
    {
        "context": (
            "中芯国际2023年全年收入为452.50亿元，毛利率为21.9%。"
            "第四季度收入为121.31亿元，毛利率为16.4%。"
            "全年资本开支为74.67亿美元，折旧为37.26亿美元。"
            "12英寸晶圆月产能达到805,700片，8英寸晶圆等值月产能约326,350片。"
        ),
        "question": "中芯国际第四季度毛利率相比全年水平表现如何？",
        "answer": "第四季度毛利率16.4%低于全年的21.9%，下滑了5.5个百分点，盈利能力明显走弱",
    },
    {
        "context": (
            "2023年中国GDP总量为126.06万亿元，按不变价格计算比上年增长5.2%。"
            "全年人均国内生产总值89358元，比上年增长5.4%。"
            "全年居民消费价格（CPI）比上年上涨0.2%。"
            "全年城镇新增就业1244万人，年末城镇调查失业率为5.1%。"
        ),
        "question": "从CPI和GDP数据来看，2023年中国经济有什么特征？",
        "answer": "GDP增长5.2%但CPI仅上涨0.2%，经济复苏态势良好但存在通缩压力",
    },
    {
        "context": (
            "比亚迪2023年全年汽车销量达302.44万辆，同比增长61.9%。"
            "其中纯电动车型销量为157.48万辆，插电式混合动力车型销量为143.81万辆。"
            "公司全年营业收入6023.15亿元，净利润300.41亿元，同比增长80.72%。"
            "研发投入达到395.75亿元，占营收比例为6.57%。"
        ),
        "question": "比亚迪的纯电动和插混车型哪个卖得更好？差距有多大？",
        "answer": "纯电动车型销量157.48万辆略高于插混的143.81万辆，领先约13.67万辆",
    },
    {
        "context": (
            "招商银行2023年实现营业收入3391.23亿元，同比下降1.64%。"
            "净利润1466.02亿元，同比增长6.22%。不良贷款率0.95%，拨备覆盖率437.70%。"
            "零售金融业务收入1903.39亿元，占总营收的56.13%。"
            "管理零售客户总资产达13.32万亿元，较上年末增长8.93%。"
        ),
        "question": "招商银行的资产质量和风险抵补能力如何？",
        "answer": "不良贷款率仅0.95%且拨备覆盖率高达437.70%，资产质量优良，风险抵补能力充足",
    },
]


def make_temp_data(tmpdir: str):
    """Write the 10 QA pairs as both NTP jsonl and QA json."""
    ntp_path = os.path.join(tmpdir, "ntp_overfit.jsonl")
    qa_path = os.path.join(tmpdir, "qa_overfit.json")

    with open(ntp_path, "w", encoding="utf-8") as f:
        for item in FINANCIAL_QA:
            # NTP data: just need the context text
            f.write(json.dumps({"text": item["context"]}, ensure_ascii=False) + "\n")

    with open(qa_path, "w", encoding="utf-8") as f:
        json.dump(FINANCIAL_QA, f, ensure_ascii=False, indent=2)

    return ntp_path, qa_path


class QADatasetWithEOS(torch.utils.data.Dataset):
    """QA dataset that appends EOS token to answers so the model learns to stop."""

    def __init__(self, data, tokenizer, max_doc_tokens=256,
                 max_question_tokens=64, max_answer_tokens=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_doc_tokens = max_doc_tokens
        self.max_question_tokens = max_question_tokens
        self.max_answer_tokens = max_answer_tokens
        self.eos_id = tokenizer.eos_token_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        doc = self.tokenizer(
            item["context"], truncation=True, max_length=self.max_doc_tokens,
            return_tensors="pt", padding=False,
        )
        question = self.tokenizer(
            item["question"], truncation=True, max_length=self.max_question_tokens,
            return_tensors="pt", padding=False,
        )
        answer = self.tokenizer(
            item["answer"], truncation=True,
            max_length=self.max_answer_tokens - 1,  # leave room for EOS
            return_tensors="pt", padding=False,
        )

        # Append EOS to answer so model learns to stop generating
        answer_ids = answer["input_ids"].squeeze(0)
        eos = torch.tensor([self.eos_id], dtype=answer_ids.dtype)
        answer_ids = torch.cat([answer_ids, eos])

        return {
            "doc_input_ids": doc["input_ids"].squeeze(0),
            "q_input_ids": question["input_ids"].squeeze(0),
            "answer_ids": answer_ids,
            "answer_labels": answer_ids,
            "answer_text": item["answer"],
        }


def build_config(ntp_steps: int, qa_steps: int) -> DeepCompressorConfig:
    """Build a minimal config for fast overfit testing on MacBook."""
    return DeepCompressorConfig(
        qwen=QwenConfig(
            model_name_or_path="models/Qwen3-0.6B",
            hidden_size=1024,
            num_hidden_layers=28,
            vocab_size=151936,
            max_doc_tokens=256,
            max_question_tokens=64,
            max_answer_tokens=64,
        ),
        finbert=FinBERTConfig(enabled=False),
        perceiver=PerceiverConfig(
            perceiver_dim=1024,
            num_queries=64,
            num_heads=16,
            head_dim=64,
            stage_a_cross_layers=2,
            stage_a_self_layers=2,
            stage_b_layers=2,
            stage_c_cross_layers=2,
            stage_c_self_layers=4,
            ff_mult=4,
            anchor_score_scale_init=1.0,
            dropout=0.0,   # no dropout for overfit test
        ),
        projection=ProjectionConfig(
            down_hidden=768,
            up_hidden=768,
            dropout=0.0,
        ),
        loss=LossConfig(
            kl_temperature=2.0,
            hidden_distill_ramp_steps=200,
            hidden_distill_layers=[7, 14, 21, 27],
            qa_ce_weight=1.0,
            kl_weight=0.0,         # no distillation for overfit test
            hidden_mse_weight=0.0,
            anchor_recon_weight=0.0,
        ),
        training=TrainingConfig(
            stage=1,
            learning_rate=1e-3,     # high LR for fast overfit
            batch_size=2,
            gradient_accumulation_steps=1,
            max_steps=ntp_steps,
            warmup_steps=10,
            weight_decay=0.0,       # no regularization for overfit
            max_grad_norm=1.0,
            scheduler="cosine",
            seed=42,
            log_every=20,
            eval_every=9999,
            save_every=9999,
            output_dir="outputs/overfit_test",
            ntp_segment_len=64,
            gradient_checkpointing=True,
            mixed_precision="no",
        ),
    )


def train_loop(model, dataloader, config, device, max_steps, stage_name, lr=1e-3,
               eval_callback=None, eval_every=None):
    """Minimal training loop without accelerate (simpler for single-device overfit).

    Args:
        eval_callback: callable(step, loss) -> snapshot_dict, called at eval points.
                       The model is temporarily set to eval mode during callback.
        eval_every: evaluate every N steps. If None, no mid-training evaluation.
    """
    model.train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.0,
    )
    # Linear warmup then constant
    warmup_steps = min(10, max_steps // 10)

    step = 0
    epoch = 0
    losses = []
    snapshots = []  # list of (step, loss, eval_results)
    t0 = time.time()

    # Evaluate at step 0 (before any training) if callback provided
    if eval_callback is not None:
        model.eval()
        snap = eval_callback(0, float("inf"))
        snapshots.append(snap)
        model.train()

    while step < max_steps:
        epoch += 1
        for batch in dataloader:
            # Move batch to device
            batch_dev = {}
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch_dev[k] = v.to(device)
                else:
                    batch_dev[k] = v

            if stage_name == "ntp":
                out = model(
                    mode="ntp",
                    doc_input_ids=batch_dev["doc_input_ids"],
                    doc_attention_mask=batch_dev["doc_attention_mask"],
                    segment_ids=batch_dev["segment_ids"],
                    segment_attention_mask=batch_dev["segment_attention_mask"],
                    segment_labels=batch_dev["segment_labels"],
                )
            else:
                out = model(
                    mode="qa",
                    doc_input_ids=batch_dev["doc_input_ids"],
                    doc_attention_mask=batch_dev["doc_attention_mask"],
                    q_input_ids=batch_dev["q_input_ids"],
                    q_attention_mask=batch_dev["q_attention_mask"],
                    answer_ids=batch_dev["answer_ids"],
                    answer_attention_mask=batch_dev["answer_attention_mask"],
                    answer_labels=batch_dev["answer_labels"],
                    global_step=step,
                )

            loss = out["total"]
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)

            # Simple warmup
            if step < warmup_steps:
                for pg in optimizer.param_groups:
                    pg["lr"] = lr * (step + 1) / warmup_steps

            optimizer.step()
            optimizer.zero_grad()

            loss_val = loss.item()
            losses.append(loss_val)
            step += 1

            if step % 20 == 0 or step == max_steps:
                elapsed = time.time() - t0
                recent = sum(losses[-20:]) / len(losses[-20:])
                logger.info(
                    f"  [{stage_name}] step {step}/{max_steps}  "
                    f"loss={recent:.4f}  elapsed={elapsed:.0f}s"
                )

            # Mid-training evaluation
            if (eval_callback is not None and eval_every is not None
                    and step % eval_every == 0 and step < max_steps):
                model.eval()
                snap = eval_callback(step, loss_val)
                snapshots.append(snap)
                model.train()

            if step >= max_steps:
                break

    # Final evaluation at last step
    if eval_callback is not None:
        recent_loss = sum(losses[-10:]) / max(len(losses[-10:]), 1)
        model.eval()
        snap = eval_callback(step, recent_loss)
        snapshots.append(snap)
        model.train()

    return losses, snapshots


@torch.no_grad()
def evaluate(model, qa_data, tokenizer, device):
    """Generate answers and compare with ground truth."""
    model.eval()
    results = []

    for item in qa_data:
        # Tokenize
        doc_tok = tokenizer(
            item["context"], truncation=True, max_length=256,
            return_tensors="pt", padding=False,
        )
        q_tok = tokenizer(
            item["question"], truncation=True, max_length=64,
            return_tensors="pt", padding=False,
        )

        doc_ids = doc_tok["input_ids"].to(device)
        doc_mask = doc_tok["attention_mask"].to(device)
        q_ids = q_tok["input_ids"].to(device)
        q_mask = q_tok["attention_mask"].to(device)

        # Compress
        byte_array = model.encode_document(doc_ids, doc_mask)
        queries = model.encode_question(q_ids, q_mask)
        latent = model.compress(queries, byte_array, byte_mask=doc_mask)
        prefix = model.up_mlp(latent)

        # Generate
        gen_ids = model.generate_answer(
            prefix, q_ids, q_mask, tokenizer=tokenizer, max_new_tokens=64,
        )
        pred = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

        gold = item["answer"]
        em = compute_exact_match(pred, gold)
        f1 = compute_f1(pred, gold)

        results.append({
            "question": item["question"],
            "gold": gold,
            "pred": pred,
            "em": em,
            "f1": f1,
        })

    return results


def make_eval_callback(model, qa_data, tokenizer, device):
    """Create a callback that evaluates the model and returns a snapshot.

    Returns a closure suitable as eval_callback for train_loop.
    """
    def callback(step, loss):
        results = evaluate(model, qa_data, tokenizer, device)
        total_em = sum(r["em"] for r in results) / len(results)
        total_f1 = sum(r["f1"] for r in results) / len(results)

        # Print a brief summary line
        preds_preview = [r["pred"][:15] for r in results[:3]]
        logger.info(
            f"  [eval @ step {step}]  EM={total_em:.0%}  F1={total_f1:.2%}  "
            f"loss={loss:.4f}  preview: {preds_preview}"
        )

        return {
            "step": step,
            "loss": loss,
            "em": total_em,
            "f1": total_f1,
            "results": results,  # full per-sample predictions
        }

    return callback


def print_report(results, ntp_losses, qa_losses, elapsed, snapshots=None):
    """Print a detailed report with pass/fail verdict and answer progression."""
    print("\n" + "=" * 70)
    print("  OVERFIT TEST REPORT — DeepCompressor Financial QA Feasibility")
    print("=" * 70)

    # Training summary
    ntp_final = sum(ntp_losses[-10:]) / max(len(ntp_losses[-10:]), 1)
    ntp_init = sum(ntp_losses[:10]) / max(len(ntp_losses[:10]), 1)
    qa_final = sum(qa_losses[-10:]) / max(len(qa_losses[-10:]), 1)
    qa_init = sum(qa_losses[:10]) / max(len(qa_losses[:10]), 1)
    print(f"\n  NTP final loss:  {ntp_final:.4f}  (initial: {ntp_init:.4f})")
    print(f"  QA  final loss:  {qa_final:.4f}  (initial: {qa_init:.4f})")
    print(f"  Total time:      {elapsed:.0f}s")

    # ── Answer Progression During Training ──
    if snapshots and len(snapshots) > 1:
        print("\n" + "=" * 70)
        print("  ANSWER PROGRESSION DURING QA TRAINING")
        print("=" * 70)

        # Summary table: step / loss / EM / F1
        print(f"\n  {'Step':<8} {'Loss':<10} {'EM':<8} {'F1':<8}")
        print("  " + "-" * 34)
        for snap in snapshots:
            loss_str = f"{snap['loss']:.4f}" if snap['loss'] != float("inf") else "  -"
            print(f"  {snap['step']:<8} {loss_str:<10} "
                  f"{snap['em']:.0%}{'':>4} {snap['f1']:.2%}")

        # Per-sample progression table
        n_samples = len(snapshots[0]["results"])
        print(f"\n  Per-sample answer evolution ({n_samples} samples):")

        for i in range(n_samples):
            gold = snapshots[0]["results"][i]["gold"]
            question_short = snapshots[0]["results"][i]["question"]
            if len(question_short) > 30:
                question_short = question_short[:28] + ".."
            print(f"\n  Sample {i+1}: {question_short}")
            print(f"  {'Gold':<10}: {gold}")
            for snap in snapshots:
                r = snap["results"][i]
                step = snap["step"]
                pred = r["pred"] if r["pred"] else "(empty)"
                em_mark = "Y" if r["em"] > 0 else " "
                print(f"  Step {step:<5}: {pred:<30} [{em_mark}]")

    # Detailed results (final evaluation)
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    print(f"\n  {'#':<3} {'EM':<4} {'F1':<6} {'Gold':<20} {'Pred':<20}")
    print("-" * 70)

    total_em = 0
    total_f1 = 0
    numeric_em = 0
    numeric_count = 0

    for i, r in enumerate(results):
        em_str = "Y" if r["em"] > 0 else "N"
        gold_short = r["gold"][:18]
        pred_short = r["pred"][:18] if r["pred"] else "(empty)"
        print(f"  {i+1:<3} {em_str:<4} {r['f1']:.2f}   {gold_short:<20} {pred_short:<20}")

        total_em += r["em"]
        total_f1 += r["f1"]

        # Check if answer contains numbers
        if re.search(r"\d", r["gold"]):
            numeric_count += 1
            numeric_em += r["em"]

    n = len(results)
    avg_em = total_em / n
    avg_f1 = total_f1 / n
    numeric_em_rate = numeric_em / numeric_count if numeric_count > 0 else 0

    print("-" * 70)
    print(f"\n  Overall EM:     {avg_em:.0%} ({int(total_em)}/{n})")
    print(f"  Overall F1:     {avg_f1:.2%}")
    print(f"  Numeric EM:     {numeric_em_rate:.0%} ({int(numeric_em)}/{numeric_count})")

    # Verdict
    print("\n" + "=" * 70)
    if avg_em >= 0.7:
        print("  VERDICT: PASS")
        print("  The architecture CAN memorize financial QA with exact numbers.")
        print("  Compression pipeline preserves numerical information.")
    elif avg_f1 >= 0.5:
        print("  VERDICT: PARTIAL PASS")
        print("  F1 is decent but exact match is low — numbers are approximately")
        print("  preserved but not exactly. May need more training or tuning.")
    else:
        print("  VERDICT: FAIL")
        print("  The model cannot recover numerical answers from compressed prefix.")
        print("  The architecture may be fundamentally limited for this task.")
    print("=" * 70)

    # Detailed failures for analysis
    failures = [r for r in results if r["em"] == 0]
    if failures:
        print("\n  FAILURE ANALYSIS:")
        for r in failures:
            print(f"    Q: {r['question']}")
            print(f"    Gold: {r['gold']}")
            print(f"    Pred: {r['pred']}")
            print()


def main():
    parser = argparse.ArgumentParser(description="Overfit test for DeepCompressor")
    parser.add_argument("--ntp_steps", type=int, default=300,
                        help="NTP pretraining steps (Stage 1)")
    parser.add_argument("--qa_steps", type=int, default=2000,
                        help="QA overfit steps (Stage 2)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--skip_ntp", action="store_true",
                        help="Skip Stage 1 NTP (use random prefix)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Load checkpoint before QA stage")
    parser.add_argument("--eval_every", type=int, default=None,
                        help="Evaluate on all QA samples every N QA steps "
                             "(default: auto = qa_steps/5)")
    args = parser.parse_args()

    # Auto-compute eval interval: ~5 snapshots during QA training
    if args.eval_every is None:
        args.eval_every = max(50, args.qa_steps // 5)

    t_start = time.time()

    # ── Setup ──
    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    logger.info(f"Device: {device}")

    config = build_config(args.ntp_steps, args.qa_steps)
    tokenizer = AutoTokenizer.from_pretrained(config.qwen.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading Qwen3-0.6B model...")
    model = DeepCompressor(config)

    if args.checkpoint:
        weights = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(weights, strict=False)
        logger.info(f"Loaded checkpoint: {args.checkpoint}")

    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {trainable:,} trainable / {total:,} total "
                f"({100*trainable/total:.1f}%)")

    # ── Prepare data ──
    with tempfile.TemporaryDirectory() as tmpdir:
        ntp_path, qa_path = make_temp_data(tmpdir)

        from deep_compressor.data import NTPDataset, QADataset, PaddingCollator
        from torch.utils.data import DataLoader

        collator = PaddingCollator(pad_token_id=tokenizer.pad_token_id)

        # ── Stage 1: NTP Overfit ──
        ntp_losses = [0.0] * 10  # dummy if skipped
        ntp_snapshots = []
        if not args.skip_ntp:
            logger.info(f"=== Stage 1: NTP Overfit ({args.ntp_steps} steps) ===")
            ntp_ds = NTPDataset(
                ntp_path, tokenizer,
                max_doc_tokens=256, segment_len=64,
            )
            ntp_loader = DataLoader(
                ntp_ds, batch_size=2, shuffle=True,
                collate_fn=collator, num_workers=0,
            )
            ntp_losses, ntp_snapshots = train_loop(
                model, ntp_loader, config, device,
                max_steps=args.ntp_steps, stage_name="ntp", lr=args.lr,
            )
            logger.info("Stage 1 complete.")
        else:
            logger.info("Skipping Stage 1 NTP (--skip_ntp)")

        # ── Stage 2: QA Overfit with mid-training evaluation ──
        logger.info(f"=== Stage 2: QA Overfit ({args.qa_steps} steps, "
                    f"eval every {args.eval_every} steps) ===")
        qa_ds = QADatasetWithEOS(
            FINANCIAL_QA, tokenizer,
            max_doc_tokens=256, max_question_tokens=64,
            max_answer_tokens=64,
        )
        qa_loader = DataLoader(
            qa_ds, batch_size=2, shuffle=True,
            collate_fn=collator, num_workers=0,
        )

        eval_cb = make_eval_callback(model, FINANCIAL_QA, tokenizer, device)
        qa_losses, qa_snapshots = train_loop(
            model, qa_loader, config, device,
            max_steps=args.qa_steps, stage_name="qa", lr=args.lr,
            eval_callback=eval_cb, eval_every=args.eval_every,
        )
        logger.info("Stage 2 complete.")

    elapsed = time.time() - t_start

    # The last snapshot from qa_snapshots IS the final evaluation
    final_results = qa_snapshots[-1]["results"] if qa_snapshots else []
    print_report(final_results, ntp_losses, qa_losses, elapsed,
                 snapshots=qa_snapshots)


if __name__ == "__main__":
    main()
