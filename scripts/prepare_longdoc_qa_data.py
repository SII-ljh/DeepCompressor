#!/usr/bin/env python3
"""Download and prepare LONG-DOCUMENT QA datasets for Deep Compressor training.

Targets documents in the 4K–64K token range, suitable for training the
compression model on genuinely long text.

Datasets:
  1. LongBench v1 (THUDM)     — 8 QA subsets, 5K–25K word contexts, EN+ZH
  2. LongBench v2 (THUDM)     — 503 MC QA, 8K–2M word contexts, EN+ZH
  3. NarrativeQA (deepmind)    — 46K QA, full stories ~60K tokens, EN
  4. NarrativeQA (SCROLLS)     — same stories, SCROLLS packaging
  5. QuALITY (emozilla)        — ~2.5K MC→free-form QA, ~5K tokens, EN
  6. QuALITY (SCROLLS)         — same, SCROLLS packaging
  7. Qasper (SCROLLS)          — scientific paper QA
  8. L-Eval Generation         — financial/legal/scientific QA, 3K–200K, EN

Output: [{"context": str, "question": str, "answer": str, "source": str}]
Compatible with existing QADataset without model code changes.

Usage:
    python scripts/prepare_longdoc_qa_data.py                    # full download
    python scripts/prepare_longdoc_qa_data.py --test             # 50 samples/dataset
    python scripts/prepare_longdoc_qa_data.py --min_tokens 8192  # only 8K+ docs
    python scripts/prepare_longdoc_qa_data.py --no-cache         # force re-download
"""

import argparse
import hashlib
import json
import os
import random
import sys
import time
import traceback
import unicodedata
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from io import BytesIO
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / ".longdoc_cache"

HF_TOKEN = os.environ.get("HF_TOKEN", None)


# ═══════════════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════════════

def _normalize_text(text: str) -> str:
    text = text.strip().lower()
    text = unicodedata.normalize("NFKC", text)
    text = " ".join(text.split())
    return text


class _Timer:
    def __init__(self, label):
        self.label = label
        self.t0 = None
    def __enter__(self):
        self.t0 = time.time()
        return self
    def __exit__(self, *args):
        print(f"  [{self.label}] {time.time() - self.t0:.1f}s", flush=True)


def _iter_progress(iterable, desc="Processing", every=5000):
    count = 0
    t0 = time.time()
    last_print = t0
    for item in iterable:
        count += 1
        now = time.time()
        if count % every == 0 or (now - last_print) >= 30:
            elapsed = now - t0
            rate = count / max(elapsed, 0.01)
            print(f"    ... {desc}: {count:,} ({elapsed:.0f}s, {rate:.0f}/s)", flush=True)
            last_print = now
        yield item
    elapsed = time.time() - t0
    print(f"    ... {desc}: {count:,} total ({elapsed:.0f}s)", flush=True)


# ═══════════════════════════════════════════════════════════════════════
#  Cache
# ═══════════════════════════════════════════════════════════════════════

def _cache_path(name): return CACHE_DIR / f"{name}.json"
def _cache_exists(name): return _cache_path(name).exists()

def _cache_save(name, samples):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_cache_path(name), "w", encoding="utf-8") as f:
        json.dump(samples, f, ensure_ascii=False)
    print(f"    -> cached {len(samples):,} samples", flush=True)

def _cache_load(name):
    with open(_cache_path(name), "r", encoding="utf-8") as f:
        return json.load(f)


# ═══════════════════════════════════════════════════════════════════════
#  HuggingFace Hub helpers
# ═══════════════════════════════════════════════════════════════════════

def _hf_download(repo_id, filename):
    """Download a file from a HuggingFace dataset repo."""
    from huggingface_hub import hf_hub_download
    return hf_hub_download(
        repo_id=repo_id, filename=filename,
        repo_type="dataset", token=HF_TOKEN,
    )


def _hf_download_and_read_jsonl(repo_id, filename):
    """Download a JSONL file and return list of dicts."""
    path = _hf_download(repo_id, filename)
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _hf_download_zip_and_read_jsonl(repo_id, zip_filename, inner_path):
    """Download a zip, extract a JSONL file inside it, return list of dicts."""
    zip_path = _hf_download(repo_id, zip_filename)
    data = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open(inner_path) as f:
            for line in f:
                line = line.decode("utf-8").strip()
                if line:
                    data.append(json.loads(line))
    return data


# ═══════════════════════════════════════════════════════════════════════
#  Dataset downloaders
# ═══════════════════════════════════════════════════════════════════════

def download_longbench(max_n=0):
    """LongBench v1: download data.zip, extract QA subsets as JSONL.

    Fields: context, input (question), answers (list), length, dataset, language
    """
    QA_SUBSETS = [
        "narrativeqa", "multifieldqa_en", "multifieldqa_zh",
        "hotpotqa", "2wikimqa", "musique", "qasper", "dureader",
    ]
    all_samples = []

    # Download the zip once
    print("    Downloading THUDM/LongBench data.zip...", flush=True)
    zip_path = _hf_download("THUDM/LongBench", "data.zip")

    with zipfile.ZipFile(zip_path, "r") as zf:
        for subset in QA_SUBSETS:
            inner = f"data/{subset}.jsonl"
            try:
                rows = []
                with zf.open(inner) as f:
                    for line in f:
                        line = line.decode("utf-8").strip()
                        if line:
                            rows.append(json.loads(line))

                count = 0
                for row in rows:
                    context = row.get("context", "")
                    question = row.get("input", "")
                    answers = row.get("answers", [])
                    if not context or not question or not answers:
                        continue
                    answer = answers[0] if isinstance(answers, list) else str(answers)
                    all_samples.append({
                        "context": context,
                        "question": question,
                        "answer": answer,
                        "source": f"longbench_{subset}",
                    })
                    count += 1
                    if max_n and count >= max_n:
                        break
                print(f"      {subset}: {count}", flush=True)
            except Exception as e:
                print(f"      {subset}: FAILED ({e})", flush=True)

    return all_samples


def download_longbench_v2(max_n=0):
    """LongBench v2: 503 MC questions with 8K-2M word contexts."""
    from datasets import load_dataset
    samples = []
    try:
        print("    Loading THUDM/LongBench-v2...", flush=True)
        ds = load_dataset("THUDM/LongBench-v2", split="train")
        count = 0
        for row in ds:
            context = row.get("context", "")
            question = row.get("question", "")
            answer_letter = row.get("answer", "")
            if not context or not question:
                continue

            # Convert letter to answer text
            answer = answer_letter
            if answer_letter in ("A", "B", "C", "D"):
                key = f"choice_{answer_letter}"
                if row.get(key):
                    answer = row[key]

            if not answer:
                continue
            samples.append({
                "context": context,
                "question": question,
                "answer": str(answer),
                "source": "longbench_v2",
            })
            count += 1
            if max_n and count >= max_n:
                break
        print(f"      -> {count} samples", flush=True)
    except Exception as e:
        print(f"      FAILED: {e}", flush=True)
        traceback.print_exc()
    return samples


def download_narrativeqa_direct(max_n=0):
    """NarrativeQA from deepmind — full story texts (very long, ~200K tokens)."""
    from datasets import load_dataset
    samples = []
    try:
        print("    Loading deepmind/narrativeqa (train+validation)...", flush=True)
        for split in ["train", "validation"]:
            ds = load_dataset("deepmind/narrativeqa", split=split, token=HF_TOKEN)

            # Check first row for full text
            if len(ds) > 0:
                doc0 = ds[0].get("document", {})
                has_text = bool(doc0.get("text", "").strip())
                print(f"      {split}: {len(ds)} rows, full_text={has_text}", flush=True)

            count = 0
            for row in _iter_progress(ds, desc=f"narrativeqa-{split}"):
                doc = row.get("document", {})
                text = doc.get("text", "").strip()
                if not text:
                    # Fallback to summary
                    summary = doc.get("summary", "")
                    if isinstance(summary, dict):
                        text = summary.get("text", "").strip()
                    elif isinstance(summary, str):
                        text = summary.strip()
                if not text:
                    continue

                q = row.get("question", {})
                question = q.get("text", "").strip() if isinstance(q, dict) else str(q).strip()

                answers_raw = row.get("answers", [])
                if not answers_raw:
                    continue
                if isinstance(answers_raw[0], dict):
                    answer = answers_raw[0].get("text", "").strip()
                else:
                    answer = str(answers_raw[0]).strip()

                if not answer or not question:
                    continue

                samples.append({
                    "context": text,
                    "question": question,
                    "answer": answer,
                    "source": f"narrativeqa_{split}",
                })
                count += 1
                if max_n and count >= max_n:
                    break
            print(f"      {split}: {count} samples extracted", flush=True)
    except Exception as e:
        print(f"      FAILED: {e}", flush=True)
        traceback.print_exc()
    return samples


def download_scrolls_narrativeqa(max_n=0):
    """NarrativeQA via SCROLLS zip — question first, then document in 'input'."""
    samples = []
    for split in ["train", "validation"]:
        try:
            inner = f"narrative_qa/{split}.jsonl"
            print(f"    Downloading tau/scrolls narrative_qa.zip ({split})...", flush=True)
            rows = _hf_download_zip_and_read_jsonl("tau/scrolls", "narrative_qa.zip", inner)
            print(f"      {split}: {len(rows)} rows loaded", flush=True)

            count = 0
            for row in rows:
                inp = row.get("input", "")
                output = row.get("output", "")
                if not inp or not output:
                    continue

                # SCROLLS format: question comes first, then document
                # Try to separate by looking for the story start
                # The question is typically a single sentence at the beginning
                parts = inp.split("\n\n", 1)
                if len(parts) == 2 and len(parts[0]) < 500:
                    question = parts[0].strip()
                    context = parts[1].strip()
                else:
                    # Fallback: first line is question
                    lines = inp.split("\n", 1)
                    if len(lines) == 2:
                        question = lines[0].strip()
                        context = lines[1].strip()
                    else:
                        continue

                if not context or not question or len(context) < 200:
                    continue

                samples.append({
                    "context": context,
                    "question": question,
                    "answer": output.strip(),
                    "source": f"scrolls_narrativeqa_{split}",
                })
                count += 1
                if max_n and count >= max_n:
                    break
            print(f"      {split}: {count} samples", flush=True)
        except Exception as e:
            print(f"      FAILED ({split}): {e}", flush=True)
            traceback.print_exc()
    return samples


def download_quality_direct(max_n=0):
    """QuALITY — long articles, multiple-choice → free-form."""
    from datasets import load_dataset
    samples = []
    try:
        print("    Loading emozilla/quality...", flush=True)
        ds = load_dataset("emozilla/quality", split="train", token=HF_TOKEN)
        print(f"      Fields: {list(ds[0].keys())}", flush=True)

        count = 0
        for row in ds:
            article = row.get("article", "")
            if not article:
                continue

            question = row.get("question", "")
            options = row.get("options", [])
            answer_idx = row.get("answer", None)

            if not question or not options:
                continue

            # Convert index to answer text
            if isinstance(answer_idx, int) and 0 <= answer_idx < len(options):
                answer = options[answer_idx]
            elif isinstance(answer_idx, str) and answer_idx.isdigit():
                idx = int(answer_idx)
                if 0 <= idx < len(options):
                    answer = options[idx]
                else:
                    continue
            else:
                continue

            samples.append({
                "context": article,
                "question": question,
                "answer": str(answer),
                "source": "quality",
            })
            count += 1
            if max_n and count >= max_n:
                break

        print(f"      -> {count} samples", flush=True)
    except Exception as e:
        print(f"      FAILED: {e}", flush=True)
        traceback.print_exc()
    return samples


def download_scrolls_quality(max_n=0):
    """QuALITY via SCROLLS zip."""
    samples = []
    for split in ["train", "validation"]:
        try:
            inner = f"quality/{split}.jsonl"
            print(f"    Downloading tau/scrolls quality.zip ({split})...", flush=True)
            rows = _hf_download_zip_and_read_jsonl("tau/scrolls", "quality.zip", inner)
            print(f"      {split}: {len(rows)} rows", flush=True)

            count = 0
            for row in rows:
                inp = row.get("input", "")
                output = row.get("output", "")
                if not inp or not output:
                    continue

                # SCROLLS quality format: question + options at start, then article
                parts = inp.split("\n\n", 1)
                if len(parts) == 2 and len(parts[0]) < 1000:
                    question = parts[0].strip()
                    context = parts[1].strip()
                else:
                    lines = inp.split("\n", 1)
                    if len(lines) == 2:
                        question = lines[0].strip()
                        context = lines[1].strip()
                    else:
                        continue

                if not context or not question or len(context) < 200:
                    continue

                samples.append({
                    "context": context,
                    "question": question,
                    "answer": output.strip(),
                    "source": f"scrolls_quality_{split}",
                })
                count += 1
                if max_n and count >= max_n:
                    break
            print(f"      {split}: {count} samples", flush=True)
        except Exception as e:
            print(f"      FAILED ({split}): {e}", flush=True)
    return samples


def download_scrolls_qasper(max_n=0):
    """Qasper via SCROLLS zip — scientific paper QA."""
    samples = []
    for split in ["train", "validation"]:
        try:
            inner = f"qasper/{split}.jsonl"
            print(f"    Downloading tau/scrolls qasper.zip ({split})...", flush=True)
            rows = _hf_download_zip_and_read_jsonl("tau/scrolls", "qasper.zip", inner)
            print(f"      {split}: {len(rows)} rows", flush=True)

            count = 0
            for row in rows:
                inp = row.get("input", "")
                output = row.get("output", "")
                if not inp or not output:
                    continue

                # SCROLLS qasper: question first, then paper
                parts = inp.split("\n\n", 1)
                if len(parts) == 2 and len(parts[0]) < 500:
                    question = parts[0].strip()
                    context = parts[1].strip()
                else:
                    lines = inp.split("\n", 1)
                    if len(lines) == 2:
                        question = lines[0].strip()
                        context = lines[1].strip()
                    else:
                        continue

                if not context or not question or len(context) < 200:
                    continue

                samples.append({
                    "context": context,
                    "question": question,
                    "answer": output.strip(),
                    "source": f"scrolls_qasper_{split}",
                })
                count += 1
                if max_n and count >= max_n:
                    break
            print(f"      {split}: {count} samples", flush=True)
        except Exception as e:
            print(f"      FAILED ({split}): {e}", flush=True)
    return samples


def download_leval(max_n=0):
    """L-Eval Generation QA tasks — download JSONL files directly.

    Fields: input (context), instructions (list of questions), outputs (list of answers)
    """
    QA_TASKS = [
        "financial_qa", "legal_contract_qa", "scientific_qa",
        "multidoc_qa", "natural_question", "narrative_qa",
        "meeting_qa", "review_qa", "news_qa",
        "paper_assistant", "patent_qa", "tv_show_qa",
    ]

    all_samples = []
    for task in QA_TASKS:
        filename = f"LEval/Generation/{task}.jsonl"
        try:
            print(f"    Downloading L-Eval/{task}...", flush=True)
            rows = _hf_download_and_read_jsonl("L4NLP/LEval", filename)

            count = 0
            for row in rows:
                context = row.get("input", "")
                instructions = row.get("instructions", [])
                outputs = row.get("outputs", [])

                if not context or not instructions or not outputs:
                    continue

                for q, a in zip(instructions, outputs):
                    if not q or not a:
                        continue
                    all_samples.append({
                        "context": context,
                        "question": q.strip(),
                        "answer": a.strip(),
                        "source": f"leval_{task}",
                    })
                    count += 1
                    if max_n and count >= max_n:
                        break
                if max_n and count >= max_n:
                    break

            print(f"      -> {count} samples", flush=True)
        except Exception as e:
            print(f"      SKIP {task}: {e}", flush=True)

    return all_samples


# ═══════════════════════════════════════════════════════════════════════
#  Token-length filtering (parallel)
# ═══════════════════════════════════════════════════════════════════════

def _tokenize_lengths_worker(items, tokenizer_path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    return [len(tok(item["context"], return_tensors=None, padding=False,
                    truncation=False)["input_ids"]) for item in items]


def compute_token_lengths(samples, tokenizer_path, num_workers=18):
    if not samples:
        return []

    n = len(samples)
    effective = min(num_workers, max(1, n // 100))

    if effective <= 1:
        print(f"    Computing token lengths (1 worker, {n:,} samples)...", flush=True)
        return _tokenize_lengths_worker(samples, tokenizer_path)

    chunk_size = (n + effective - 1) // effective
    chunks = [samples[i:i + chunk_size] for i in range(0, n, chunk_size)]

    print(f"    Computing token lengths ({effective} workers, {n:,} samples)...", flush=True)
    t0 = time.time()
    all_lengths = [None] * len(chunks)
    worker = partial(_tokenize_lengths_worker, tokenizer_path=tokenizer_path)
    with ProcessPoolExecutor(max_workers=effective) as pool:
        futures = {pool.submit(worker, c): i for i, c in enumerate(chunks)}
        for fut in futures:
            all_lengths[futures[fut]] = fut.result()

    flat = []
    for chunk_l in all_lengths:
        flat.extend(chunk_l)
    print(f"    Done in {time.time() - t0:.1f}s", flush=True)
    return flat


# ═══════════════════════════════════════════════════════════════════════
#  Truncation for very long docs
# ═══════════════════════════════════════════════════════════════════════

def truncate_long_docs(samples, lengths, max_tokens, chars_per_token=4.0):
    """For docs exceeding max_tokens, truncate context by estimated char count.

    Returns (new_samples, needs_recount_indices).
    """
    truncated_count = 0
    for i, (s, l) in enumerate(zip(samples, lengths)):
        if l > max_tokens:
            # Estimate characters to keep
            ratio = max_tokens / l
            max_chars = int(len(s["context"]) * ratio * 0.95)  # 5% safety margin
            s["context"] = s["context"][:max_chars]
            truncated_count += 1

    if truncated_count:
        print(f"    Truncated {truncated_count:,} docs exceeding {max_tokens:,} tokens", flush=True)
    return samples


# ═══════════════════════════════════════════════════════════════════════
#  Cleaning & dedup
# ═══════════════════════════════════════════════════════════════════════

def clean_data(samples, label="data"):
    total = len(samples)
    reasons = Counter()
    cleaned = []
    for s in samples:
        ctx = (s.get("context") or "").strip()
        q = (s.get("question") or "").strip()
        a = (s.get("answer") or "").strip()
        if not ctx: reasons["missing_context"] += 1; continue
        if not q: reasons["missing_question"] += 1; continue
        if not a: reasons["missing_answer"] += 1; continue
        if len(ctx) < 50: reasons["short_context"] += 1; continue
        if len(q) < 2: reasons["short_question"] += 1; continue
        if _normalize_text(q) == _normalize_text(a): reasons["q_equals_a"] += 1; continue
        s["context"] = ctx; s["question"] = q; s["answer"] = a
        cleaned.append(s)
    removed = total - len(cleaned)
    print(f"\n  [{label}] Cleaning: {total:,} -> {len(cleaned):,} ({removed:,} removed)")
    for r, c in reasons.most_common():
        print(f"    - {r}: {c:,}")
    return cleaned


def deduplicate(samples):
    seen = set()
    unique = []
    for s in samples:
        key = hashlib.md5((s["context"][:500] + "||" + s["question"]).encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            unique.append(s)
    print(f"  Dedup: {len(samples):,} -> {len(unique):,} ({len(samples) - len(unique):,} removed)")
    return unique


# ═══════════════════════════════════════════════════════════════════════
#  Statistics
# ═══════════════════════════════════════════════════════════════════════

def print_length_stats(samples, lengths, label="data"):
    if not lengths:
        return
    arr = np.array(lengths)
    p = np.percentile(arr, [10, 25, 50, 75, 90, 95, 99])
    print(f"\n{'=' * 70}")
    print(f"  TOKEN LENGTH STATS [{label}] ({len(arr):,} samples)")
    print(f"{'=' * 70}")
    print(f"  Min:  {arr.min():>8,}    Max: {arr.max():>8,}    Mean: {arr.mean():>10.1f}")
    print(f"  P10:  {int(p[0]):>8,}    P25: {int(p[1]):>8,}    P50:  {int(p[2]):>8,}")
    print(f"  P75:  {int(p[3]):>8,}    P90: {int(p[4]):>8,}    P95:  {int(p[5]):>8,}")
    print(f"  P99:  {int(p[6]):>8,}")

    buckets = [0, 1024, 2048, 4096, 8192, 16384, 32768, 65536, float("inf")]
    labels = ["<1K", "1-2K", "2-4K", "4-8K", "8-16K", "16-32K", "32-64K", ">64K"]
    print(f"\n  Distribution:")
    for i in range(len(labels)):
        cnt = int(np.sum((arr >= buckets[i]) & (arr < buckets[i + 1])))
        pct = 100.0 * cnt / len(arr)
        bar = "█" * int(40 * cnt / max(len(arr), 1))
        print(f"    {labels[i]:>8} │{bar:<40} {cnt:>8,} ({pct:5.1f}%)")

    src_counts = Counter(s["source"] for s in samples)
    print(f"\n  Per-source:")
    for src, cnt in src_counts.most_common():
        sl = [l for s, l in zip(samples, lengths) if s["source"] == src]
        sa = np.array(sl)
        print(f"    {src:<35} n={cnt:>6,}  P50={int(np.median(sa)):>7,}  "
              f"P95={int(np.percentile(sa, 95)):>7,}  max={int(sa.max()):>7,}")


# ═══════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════

DATASETS = {
    "longbench":              download_longbench,
    "longbench_v2":           download_longbench_v2,
    "narrativeqa_direct":     download_narrativeqa_direct,
    "scrolls_narrativeqa":    download_scrolls_narrativeqa,
    "quality_direct":         download_quality_direct,
    "scrolls_quality":        download_scrolls_quality,
    "scrolls_qasper":         download_scrolls_qasper,
    "leval":                  download_leval,
}


def main():
    parser = argparse.ArgumentParser(
        description="Download long-document QA data for Deep Compressor")
    parser.add_argument("--min_tokens", type=int, default=4096,
                        help="Min doc token length (default: 4096)")
    parser.add_argument("--max_tokens", type=int, default=65536,
                        help="Max doc token length (default: 65536)")
    parser.add_argument("--truncate_long", action="store_true", default=True,
                        help="Truncate docs > max_tokens instead of dropping (default: True)")
    parser.add_argument("--no_truncate", action="store_true",
                        help="Drop docs > max_tokens instead of truncating")
    parser.add_argument("--model", default="models/Qwen3-0.6B",
                        help="Tokenizer path (default: models/Qwen3-0.6B)")
    parser.add_argument("--num_workers", type=int, default=18)
    parser.add_argument("--dev_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test", action="store_true",
                        help="Test mode: 50 samples per dataset")
    parser.add_argument("--skip", nargs="*", default=[])
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--output_prefix", default="longdoc_qa")
    parser.add_argument("--hf_token", default=None,
                        help="HuggingFace token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    global HF_TOKEN
    if args.hf_token:
        HF_TOKEN = args.hf_token
    do_truncate = not args.no_truncate

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    max_n = 50 if args.test else 0

    # ── Download ──
    print("\n" + "=" * 70)
    print("  DOWNLOADING LONG-DOCUMENT QA DATASETS")
    print("=" * 70)

    all_samples = []
    skip = set(args.skip)
    only = set(args.only) if args.only else None

    for name, fn in DATASETS.items():
        if name in skip:
            print(f"\n  SKIP: {name}", flush=True); continue
        if only and name not in only:
            print(f"\n  SKIP: {name} (not in --only)", flush=True); continue

        print(f"\n  ── {name} ──", flush=True)
        if not getattr(args, 'no_cache', False) and _cache_exists(name):
            cached = _cache_load(name)
            all_samples.extend(cached)
            print(f"  CACHED: {len(cached):,} samples", flush=True)
            continue

        with _Timer(name):
            try:
                samples = fn(max_n=max_n)
            except Exception as e:
                print(f"  ERROR: {e}", flush=True)
                traceback.print_exc()
                samples = []

        if samples:
            _cache_save(name, samples)
        all_samples.extend(samples)
        print(f"  Total from {name}: {len(samples):,}", flush=True)

    print(f"\n  Raw total: {len(all_samples):,} samples")

    # ── Clean ──
    print("\n" + "=" * 70)
    print("  CLEANING")
    print("=" * 70)
    all_samples = clean_data(all_samples)
    all_samples = deduplicate(all_samples)

    if not all_samples:
        print("\n  ERROR: No samples after cleaning.")
        sys.exit(1)

    # ── Token lengths & filter ──
    print("\n" + "=" * 70)
    print(f"  TOKEN FILTERING [{args.min_tokens:,} – {args.max_tokens:,}]"
          f"  truncate={'ON' if do_truncate else 'OFF'}")
    print("=" * 70)

    lengths = compute_token_lengths(all_samples, args.model, args.num_workers)
    print_length_stats(all_samples, lengths, label="BEFORE filter")

    # Truncate long docs if requested
    if do_truncate:
        truncate_long_docs(all_samples, lengths, args.max_tokens)
        # Recompute lengths for truncated docs
        needs_recount = [i for i, l in enumerate(lengths) if l > args.max_tokens]
        if needs_recount:
            print(f"    Recomputing lengths for {len(needs_recount):,} truncated docs...", flush=True)
            recount_items = [all_samples[i] for i in needs_recount]
            new_lengths = compute_token_lengths(recount_items, args.model, args.num_workers)
            for idx, new_l in zip(needs_recount, new_lengths):
                lengths[idx] = new_l

    # Filter by range
    filtered = []
    filtered_lengths = []
    for s, l in zip(all_samples, lengths):
        if args.min_tokens <= l <= args.max_tokens:
            filtered.append(s)
            filtered_lengths.append(l)

    dropped = len(all_samples) - len(filtered)
    print(f"\n  Filtered: {len(all_samples):,} -> {len(filtered):,} ({dropped:,} dropped)")

    if not filtered:
        print(f"\n  WARNING: 0 samples in [{args.min_tokens}, {args.max_tokens}]!")
        print(f"  Saving all {len(all_samples):,} without filter...")
        filtered, filtered_lengths = all_samples, lengths

    print_length_stats(filtered, filtered_lengths, label="AFTER filter")

    # ── Split ──
    print("\n" + "=" * 70)
    print("  SPLIT TRAIN / DEV")
    print("=" * 70)
    rng = random.Random(args.seed)
    indices = list(range(len(filtered)))
    rng.shuffle(indices)
    dev_size = max(1, int(len(filtered) * args.dev_ratio))
    dev_set = set(indices[:dev_size])

    train = [filtered[i] for i in range(len(filtered)) if i not in dev_set]
    dev = [filtered[i] for i in range(len(filtered)) if i in dev_set]
    print(f"  Train: {len(train):,}    Dev: {len(dev):,}")

    # ── Save ──
    print("\n" + "=" * 70)
    print("  SAVING")
    print("=" * 70)
    train_path = DATA_DIR / f"{args.output_prefix}_train.json"
    dev_path = DATA_DIR / f"{args.output_prefix}_dev.json"

    for path, data, label in [(train_path, train, "train"), (dev_path, dev, "dev")]:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        mb = path.stat().st_size / (1024 * 1024)
        print(f"  {label}: {path.name}  ({len(data):,} samples, {mb:.1f} MB)")

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)
    print(f"  Token range: [{args.min_tokens:,}, {args.max_tokens:,}]")
    print(f"  Train: {len(train):,}    Dev: {len(dev):,}")
    print(f"  Sources:")
    for src, cnt in Counter(s["source"] for s in filtered).most_common():
        print(f"    {src:<35} {cnt:>6,}")
    print(f"\n  {train_path}\n  {dev_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
