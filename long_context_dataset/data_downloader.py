#!/usr/bin/env python3
"""
data_downloader.py — 科研数据集全量/采样下载器

支持断点续传、流式下载、HF镜像、CLI任务切换。

用法:
    # 下载全部数据集
    python data_downloader.py --task all

    # 仅下载预训练数据 (FineWeb 采样 ~2B tokens)
    python data_downloader.py --task pretrain --target_tokens 2_000_000_000

    # 仅下载 SFT 数据
    python data_downloader.py --task sft

    # 仅下载评测数据
    python data_downloader.py --task eval

    # 仅下载诊断数据
    python data_downloader.py --task diagnostic

    # 使用 HF 镜像 (中国境内推荐)
    python data_downloader.py --task all --mirror

    # 自定义输出目录
    python data_downloader.py --task all --output_dir /path/to/data
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# HF 镜像配置 (需在 import datasets 之前设置)
# ---------------------------------------------------------------------------
def setup_hf_mirror(use_mirror: bool = False):
    """设置 HuggingFace 镜像站 (中国境内加速)."""
    if use_mirror:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        print("[Mirror] 已启用 HF 镜像: https://hf-mirror.com")
    else:
        # 如果环境变量中已有镜像设置，保留用户自定义
        endpoint = os.environ.get("HF_ENDPOINT", "")
        if endpoint:
            print(f"[Mirror] 使用已有 HF_ENDPOINT: {endpoint}")


# ---------------------------------------------------------------------------
# 延迟导入 (镜像设置需在 import 前完成)
# ---------------------------------------------------------------------------
def lazy_imports():
    global datasets, load_dataset, tqdm
    import datasets as _datasets
    from datasets import load_dataset as _load_dataset
    from tqdm import tqdm as _tqdm
    datasets = _datasets
    load_dataset = _load_dataset
    tqdm = _tqdm


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------
def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_jsonl(records: list[dict], path: Path, mode: str = "w"):
    with open(path, mode, encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def append_jsonl(record: dict, f):
    f.write(json.dumps(record, ensure_ascii=False) + "\n")


def count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def count_tokens_in_jsonl(path: Path, text_key: str = "text") -> int:
    """粗略估算 JSONL 中的 token 数 (按空格+标点分词, 1 char ≈ 0.6 token for中文)."""
    if not path.exists():
        return 0
    total = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                text = obj.get(text_key, "")
                # 粗略估算: 英文按空格计, 中文每字算1.5 token, 取平均 ~0.75 token/char
                total += max(len(text.split()), int(len(text) * 0.75))
            except json.JSONDecodeError:
                continue
    return total


def format_tokens(n: int) -> str:
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def save_dataset_to_disk(ds, output_dir: Path, name: str):
    """将 HF Dataset 保存到磁盘 (Arrow 格式, 支持断点续传)."""
    save_path = output_dir / name
    if save_path.exists() and (save_path / "dataset_info.json").exists():
        print(f"  [Skip] {name} 已存在于 {save_path}")
        return
    print(f"  [Save] {name} → {save_path}")
    ds.save_to_disk(str(save_path))
    print(f"  [Done] {name} ({len(ds)} samples)")


def save_dataset_as_jsonl(ds, output_path: Path, desc: str = ""):
    """将 HF Dataset 保存为 JSONL."""
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"  [Skip] {output_path.name} 已存在 ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
        return
    print(f"  [Save] {desc} → {output_path}")
    records = []
    for item in tqdm(ds, desc=desc, leave=False):
        records.append(dict(item))
    write_jsonl(records, output_path)
    print(f"  [Done] {output_path.name} ({len(records)} samples, {output_path.stat().st_size / 1024 / 1024:.1f} MB)")


# ===========================================================================
# Task 1: Pre-train — FineWeb 流式采样
# ===========================================================================
def download_pretrain(output_dir: Path, target_tokens: int = 2_000_000_000, seed: int = 42):
    """
    流式采样 FineWeb sample-10BT, 目标 ~2B tokens.

    策略: 流式遍历 + 按概率采样 (sample_rate ≈ target/total).
    FineWeb sample-10BT 约 10B tokens, 采样率 ≈ 20%.
    支持断点续传: 记录已采样 token 数到 checkpoint 文件.
    """
    pretrain_dir = ensure_dir(output_dir / "pretrain")
    output_path = pretrain_dir / "fineweb_sampled.jsonl"
    ckpt_path = pretrain_dir / ".fineweb_checkpoint.json"

    # 断点续传: 读取 checkpoint
    start_idx = 0
    accumulated_tokens = 0
    total_seen = 0
    if ckpt_path.exists():
        with open(ckpt_path, "r") as f:
            ckpt = json.load(f)
        start_idx = ckpt.get("total_seen", 0)
        accumulated_tokens = ckpt.get("accumulated_tokens", 0)
        total_seen = start_idx
        print(f"  [Resume] 从 checkpoint 恢复: 已处理 {total_seen:,} 条, "
              f"已采样 {format_tokens(accumulated_tokens)} tokens")
        if accumulated_tokens >= target_tokens:
            print(f"  [Skip] 已达到目标 {format_tokens(target_tokens)} tokens")
            return

    # FineWeb sample-10BT ≈ 15M docs, ~10B tokens → sample_rate ≈ target/10B
    total_pool_tokens = 10_000_000_000
    sample_rate = min(1.0, target_tokens / total_pool_tokens * 1.2)  # 1.2x 超采防不足

    print(f"\n{'='*70}")
    print(f"[Pretrain] FineWeb sample-10BT 流式采样")
    print(f"  目标: {format_tokens(target_tokens)} tokens")
    print(f"  采样率: {sample_rate:.2%}")
    print(f"  输出: {output_path}")
    print(f"{'='*70}")

    rng = random.Random(seed)
    ds = load_dataset(
        "HuggingFaceFW/fineweb",
        name="sample-10BT",
        split="train",
        streaming=True,
    )

    file_mode = "a" if start_idx > 0 else "w"
    save_interval = 10_000  # 每 1w 条保存一次 checkpoint
    sampled_count = count_lines(output_path) if start_idx > 0 else 0

    pbar = tqdm(
        desc=f"FineWeb → {format_tokens(accumulated_tokens)}/{format_tokens(target_tokens)}",
        unit=" docs",
        total=None,
    )

    try:
        with open(output_path, file_mode, encoding="utf-8") as f:
            for i, example in enumerate(ds):
                if i < start_idx:
                    # 跳过已处理的 (流式无法 seek, 只能快进)
                    if i % 500_000 == 0 and i > 0:
                        pbar.set_description(f"[Fast-forward] {i:,}/{start_idx:,}")
                    continue

                total_seen = i + 1
                pbar.update(1)

                # 按概率采样
                if rng.random() > sample_rate:
                    continue

                text = example.get("text", "")
                if not text or len(text) < 100:
                    continue

                # 估算 token 数 (英文 ~0.75 token/char)
                est_tokens = max(len(text.split()), int(len(text) * 0.75))
                accumulated_tokens += est_tokens
                sampled_count += 1

                append_jsonl({"text": text, "id": example.get("id", str(i))}, f)

                # 更新进度条
                pbar.set_description(
                    f"FineWeb | sampled {sampled_count:,} | "
                    f"{format_tokens(accumulated_tokens)}/{format_tokens(target_tokens)}"
                )

                # 定期 checkpoint
                if sampled_count % save_interval == 0:
                    f.flush()
                    with open(ckpt_path, "w") as cf:
                        json.dump({
                            "total_seen": total_seen,
                            "accumulated_tokens": accumulated_tokens,
                            "sampled_count": sampled_count,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        }, cf, indent=2)

                # 达到目标
                if accumulated_tokens >= target_tokens:
                    print(f"\n  [Done] 达到目标! 采样 {sampled_count:,} 条, "
                          f"估计 {format_tokens(accumulated_tokens)} tokens")
                    break
    except KeyboardInterrupt:
        print(f"\n  [Interrupt] 用户中断, 保存 checkpoint...")
    finally:
        pbar.close()
        # 保存最终 checkpoint
        with open(ckpt_path, "w") as cf:
            json.dump({
                "total_seen": total_seen,
                "accumulated_tokens": accumulated_tokens,
                "sampled_count": sampled_count,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "completed": accumulated_tokens >= target_tokens,
            }, cf, indent=2)

    file_size = output_path.stat().st_size / (1024 ** 3)
    print(f"  [Summary] {output_path}: {sampled_count:,} docs, "
          f"{format_tokens(accumulated_tokens)} tokens, {file_size:.2f} GB")


# ===========================================================================
# Task 2: SFT — 长上下文指令 & 长文摘要
# ===========================================================================
def download_sft(output_dir: Path):
    sft_dir = ensure_dir(output_dir / "sft")

    print(f"\n{'='*70}")
    print(f"[SFT] 下载长上下文 SFT 数据集")
    print(f"  输出目录: {sft_dir}")
    print(f"{'='*70}")

    # --- LongAlpaca-12k ---
    print("\n  [1/2] Yukang/LongAlpaca-12k (长指令)")
    output_path = sft_dir / "longalpaca_12k.jsonl"
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"    [Skip] 已存在 ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
    else:
        ds = load_dataset("Yukang/LongAlpaca-12k", split="train")
        save_dataset_as_jsonl(ds, output_path, desc="LongAlpaca-12k")

    # --- BookSum ---
    print("\n  [2/2] kmfoda/booksum (长文摘要)")
    for split in ["train", "validation", "test"]:
        output_path = sft_dir / f"booksum_{split}.jsonl"
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"    [Skip] booksum_{split} 已存在 ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")
            continue
        ds = load_dataset("kmfoda/booksum", split=split)
        save_dataset_as_jsonl(ds, output_path, desc=f"BookSum/{split}")

    print(f"\n  [SFT Done]")


# ===========================================================================
# Task 3: Eval — 长文评测基准
# ===========================================================================
LONGBENCH_CONFIGS = [
    "narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh",
    "hotpotqa", "2wikimqa", "musique", "dureader", "gov_report",
    "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum",
    "lsht", "passage_count", "passage_retrieval_en",
    "passage_retrieval_zh", "lcc", "repobench-p",
]

RULER_CONFIGS = [
    "cwe_4k", "cwe_8k",
    "niah_multikey_1_4k", "niah_multikey_1_8k",
    "qa_2_4k", "qa_2_8k",
    "vt_4k", "vt_8k",
]


def download_eval(output_dir: Path):
    eval_dir = ensure_dir(output_dir / "eval")

    print(f"\n{'='*70}")
    print(f"[Eval] 下载评测基准数据集")
    print(f"  输出目录: {eval_dir}")
    print(f"{'='*70}")

    # --- LongBench ---
    print("\n  [1/3] THUDM/LongBench (全面评测, 21 子集)")
    lb_dir = ensure_dir(eval_dir / "longbench")
    for cfg in tqdm(LONGBENCH_CONFIGS, desc="LongBench configs"):
        output_path = lb_dir / f"{cfg}.jsonl"
        if output_path.exists() and output_path.stat().st_size > 0:
            continue
        try:
            ds = load_dataset("THUDM/LongBench", cfg, split="test", trust_remote_code=True)
            records = [dict(item) for item in ds]
            write_jsonl(records, output_path)
        except Exception as e:
            print(f"\n    [Error] LongBench/{cfg}: {e}")
    print(f"    [Done] LongBench: {sum(1 for f in lb_dir.glob('*.jsonl'))} configs 已保存")

    # --- RULER ---
    print("\n  [2/3] rbiswasfc/ruler (长上下文压力测试)")
    ruler_dir = ensure_dir(eval_dir / "ruler")
    for cfg in tqdm(RULER_CONFIGS, desc="RULER configs"):
        output_path = ruler_dir / f"{cfg}.jsonl"
        if output_path.exists() and output_path.stat().st_size > 0:
            continue
        try:
            ds = load_dataset("rbiswasfc/ruler", cfg, split="validation")
            records = [dict(item) for item in ds]
            write_jsonl(records, output_path)
        except Exception as e:
            print(f"\n    [Error] RULER/{cfg}: {e}")
    print(f"    [Done] RULER: {sum(1 for f in ruler_dir.glob('*.jsonl'))} configs 已保存")

    # --- HotpotQA ---
    print("\n  [3/3] hotpotqa/hotpot_qa (多跳推理)")
    hotpot_dir = ensure_dir(eval_dir / "hotpotqa")
    for cfg in ["distractor", "fullwiki"]:
        for split in ["train", "validation"]:
            output_path = hotpot_dir / f"{cfg}_{split}.jsonl"
            if output_path.exists() and output_path.stat().st_size > 0:
                print(f"    [Skip] {cfg}/{split} 已存在")
                continue
            try:
                ds = load_dataset("hotpotqa/hotpot_qa", cfg, split=split)
                save_dataset_as_jsonl(ds, output_path, desc=f"HotpotQA/{cfg}/{split}")
            except Exception as e:
                print(f"    [Error] HotpotQA/{cfg}/{split}: {e}")

    print(f"\n  [Eval Done]")


# ===========================================================================
# Task 4: Diagnostic — 幻觉检测
# ===========================================================================
FAITHEVAL_VARIANTS = [
    ("Salesforce/FaithEval-unanswerable-v1.0", "faitheval_unanswerable.jsonl"),
    ("Salesforce/FaithEval-inconsistent-v1.0", "faitheval_inconsistent.jsonl"),
    ("Salesforce/FaithEval-counterfactual-v1.0", "faitheval_counterfactual.jsonl"),
]


def download_diagnostic(output_dir: Path):
    diag_dir = ensure_dir(output_dir / "diagnostic")

    print(f"\n{'='*70}")
    print(f"[Diagnostic] 下载诊断数据集 (FaithEval)")
    print(f"  输出目录: {diag_dir}")
    print(f"{'='*70}")

    for repo_id, filename in FAITHEVAL_VARIANTS:
        output_path = diag_dir / filename
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"  [Skip] {filename} 已存在 ({output_path.stat().st_size / 1024:.1f} KB)")
            continue
        print(f"  [Download] {repo_id}")
        try:
            ds = load_dataset(repo_id, split="test")
            save_dataset_as_jsonl(ds, output_path, desc=filename)
        except Exception as e:
            print(f"  [Error] {repo_id}: {e}")

    print(f"\n  [Diagnostic Done]")


# ===========================================================================
# Main
# ===========================================================================
TASK_MAP = {
    "pretrain": "Pre-train (FineWeb ~2B tokens 流式采样)",
    "sft": "SFT (LongAlpaca-12k + BookSum)",
    "eval": "Eval (LongBench + RULER + HotpotQA)",
    "diagnostic": "Diagnostic (FaithEval 幻觉检测)",
    "all": "全部数据集",
}


def main():
    parser = argparse.ArgumentParser(
        description="科研数据集下载器 — 支持流式下载、断点续传、HF镜像",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python data_downloader.py --task all                   # 下载全部
  python data_downloader.py --task pretrain --mirror      # 使用镜像下载预训练数据
  python data_downloader.py --task sft eval               # 仅下载 SFT + Eval
  python data_downloader.py --task pretrain --target_tokens 500_000_000  # 采样 0.5B tokens
        """,
    )
    parser.add_argument(
        "--task",
        nargs="+",
        choices=list(TASK_MAP.keys()),
        default=["all"],
        help="要下载的任务 (可多选, 默认: all)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data",
        help="输出根目录 (默认: ./data)",
    )
    parser.add_argument(
        "--mirror",
        action="store_true",
        help="启用 HF 镜像 (hf-mirror.com, 中国境内推荐)",
    )
    parser.add_argument(
        "--target_tokens",
        type=int,
        default=2_000_000_000,
        help="FineWeb 采样目标 token 数 (默认: 2B)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子 (默认: 42)",
    )

    args = parser.parse_args()

    # 镜像配置 (必须在 import datasets 前)
    setup_hf_mirror(args.mirror)
    lazy_imports()

    output_dir = Path(args.output_dir).resolve()
    tasks = args.task
    if "all" in tasks:
        tasks = ["pretrain", "sft", "eval", "diagnostic"]

    print(f"\n{'#'*70}")
    print(f"# 科研数据集下载器")
    print(f"# 输出目录: {output_dir}")
    print(f"# 任务: {', '.join(tasks)}")
    print(f"{'#'*70}")

    t0 = time.time()

    if "pretrain" in tasks:
        download_pretrain(output_dir, target_tokens=args.target_tokens, seed=args.seed)

    if "sft" in tasks:
        download_sft(output_dir)

    if "eval" in tasks:
        download_eval(output_dir)

    if "diagnostic" in tasks:
        download_diagnostic(output_dir)

    elapsed = time.time() - t0
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"\n{'#'*70}")
    print(f"# 全部完成! 耗时: {minutes}m {seconds}s")
    print(f"# 数据目录: {output_dir}")
    print(f"{'#'*70}")


if __name__ == "__main__":
    main()
