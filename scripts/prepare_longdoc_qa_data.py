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
import gc
import hashlib
import json
import os
import random
import resource
import signal
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


# ═══════════════════════════════════════════════════════════════════════
#  Memory monitoring
# ═══════════════════════════════════════════════════════════════════════

def _get_rss_gb():
    """Get current RSS (Resident Set Size) in GB via resource module."""
    # ru_maxrss is in bytes on macOS, KB on Linux
    usage = resource.getrusage(resource.RUSAGE_SELF)
    rss_bytes = usage.ru_maxrss
    if sys.platform == "darwin":
        return rss_bytes / (1024 ** 3)  # macOS: bytes -> GB
    else:
        return rss_bytes / (1024 ** 2)  # Linux: KB -> GB


def _get_current_rss_gb():
    """Get *current* RSS (not peak) in GB. Falls back to peak if /proc unavailable."""
    try:
        # Linux: read from /proc for current (not peak) RSS
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024 ** 2)  # KB -> GB
    except (FileNotFoundError, PermissionError):
        pass
    # macOS / fallback: use peak RSS (resource module only gives peak on macOS)
    return _get_rss_gb()


# Default memory limit (GB). Override with --max_memory_gb.
_MEMORY_LIMIT_GB = None


def _check_memory(context=""):
    """Check memory usage. Raise MemoryError with clear message if over limit."""
    if _MEMORY_LIMIT_GB is None:
        return
    rss = _get_current_rss_gb()
    if rss > _MEMORY_LIMIT_GB:
        raise MemoryError(
            f"\n{'='*70}\n"
            f"  MEMORY LIMIT EXCEEDED\n"
            f"{'='*70}\n"
            f"  Current RSS:  {rss:.2f} GB\n"
            f"  Limit:        {_MEMORY_LIMIT_GB:.1f} GB\n"
            f"  Context:      {context}\n"
            f"\n"
            f"  Suggestions:\n"
            f"    1. Use --skip {context} to skip this dataset\n"
            f"    2. Use --max_memory_gb N to raise the limit\n"
            f"    3. Use --max_context_chars 130000 to halve per-doc memory\n"
            f"    4. Close other applications to free memory\n"
            f"{'='*70}"
        )


def _sigterm_handler(signum, frame):
    """Handle SIGTERM (often sent before SIGKILL by OOM killer)."""
    rss = _get_current_rss_gb()
    peak = _get_rss_gb()
    print(
        f"\n{'='*70}\n"
        f"  SIGTERM RECEIVED (likely OOM killer)\n"
        f"{'='*70}\n"
        f"  Current/Peak RSS: {rss:.2f} / {peak:.2f} GB\n"
        f"\n"
        f"  The OS killed this process due to memory pressure.\n"
        f"  Try:\n"
        f"    --max_memory_gb 4    (abort gracefully before OOM)\n"
        f"    --max_context_chars 130000  (smaller docs)\n"
        f"    --skip scrolls_narrativeqa  (skip largest dataset)\n"
        f"{'='*70}",
        file=sys.stderr, flush=True,
    )
    sys.exit(137)


signal.signal(signal.SIGTERM, _sigterm_handler)

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


def _iter_progress(iterable, desc="Processing", every=5000, check_memory_ctx=""):
    count = 0
    t0 = time.time()
    last_print = t0
    for item in iterable:
        count += 1
        now = time.time()
        if count % every == 0 or (now - last_print) >= 30:
            elapsed = now - t0
            rate = count / max(elapsed, 0.01)
            rss = _get_current_rss_gb()
            print(f"    ... {desc}: {count:,} ({elapsed:.0f}s, {rate:.0f}/s, RSS={rss:.2f}GB)", flush=True)
            last_print = now
            if check_memory_ctx:
                _check_memory(check_memory_ctx)
        yield item
    elapsed = time.time() - t0
    rss = _get_current_rss_gb()
    print(f"    ... {desc}: {count:,} total ({elapsed:.0f}s, RSS={rss:.2f}GB)", flush=True)


# ═══════════════════════════════════════════════════════════════════════
#  Cache  (JSONL format — streaming-friendly, no full-memory load)
# ═══════════════════════════════════════════════════════════════════════

def _cache_path(name): return CACHE_DIR / f"{name}.jsonl"

def _cache_exists(name):
    # Also check legacy .json for backward compat
    return _cache_path(name).exists() or (CACHE_DIR / f"{name}.json").exists()

def _cache_save(name, samples):
    """Save list of samples as JSONL (one JSON object per line)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_cache_path(name), "w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print(f"    -> cached {len(samples):,} samples", flush=True)

def _cache_load(name):
    """Load cache, supports both JSONL (new) and JSON (legacy)."""
    jsonl_path = CACHE_DIR / f"{name}.jsonl"
    json_path = CACHE_DIR / f"{name}.json"
    path = jsonl_path if jsonl_path.exists() else json_path
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        if path.suffix == ".jsonl":
            for line in f:
                line = line.strip()
                if line:
                    samples.append(json.loads(line))
        else:
            samples = json.load(f)
    return samples


class _StreamingCacheWriter:
    """Write samples to JSONL cache one-at-a-time (constant memory)."""
    def __init__(self, name):
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self.path = _cache_path(name)
        self.f = open(self.path, "w", encoding="utf-8")
        self.count = 0

    def write(self, sample):
        self.f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        self.count += 1

    def close(self):
        self.f.close()
        print(f"    -> cached {self.count:,} samples (streaming)", flush=True)

    def __enter__(self): return self
    def __exit__(self, *a): self.close()


def _cache_count(name):
    """Count samples in cache without loading all into memory."""
    jsonl_path = CACHE_DIR / f"{name}.jsonl"
    json_path = CACHE_DIR / f"{name}.json"
    path = jsonl_path if jsonl_path.exists() else json_path
    if not path.exists():
        return 0
    if path.suffix == ".jsonl":
        count = 0
        with open(path, "r") as f:
            for line in f:
                if line.strip():
                    count += 1
        return count
    else:
        with open(path, "r") as f:
            return len(json.load(f))


def _cache_iter(name):
    """Stream samples from cache one at a time (constant memory).

    Yields dicts. Supports both JSONL and legacy JSON formats.
    """
    jsonl_path = CACHE_DIR / f"{name}.jsonl"
    json_path = CACHE_DIR / f"{name}.json"
    path = jsonl_path if jsonl_path.exists() else json_path
    if not path.exists():
        return
    if path.suffix == ".jsonl":
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
    else:
        # Legacy JSON: must load all, then yield one at a time
        with open(path, "r", encoding="utf-8") as f:
            for item in json.load(f):
                yield item


def _cache_file_size_mb(name):
    """Get cache file size in MB, or 0 if not found."""
    jsonl_path = CACHE_DIR / f"{name}.jsonl"
    json_path = CACHE_DIR / f"{name}.json"
    path = jsonl_path if jsonl_path.exists() else json_path
    if path.exists():
        return path.stat().st_size / (1024 * 1024)
    return 0


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


def download_longbench_v2(max_n=0, max_context_chars=260000):
    """LongBench v2: 503 MC questions with 8K-2M word contexts."""
    from datasets import load_dataset
    samples = []
    try:
        print("    Loading THUDM/LongBench-v2 (streaming)...", flush=True)
        ds = load_dataset("THUDM/LongBench-v2", split="train", streaming=True)
        count = 0
        for row in ds:
            context = row.get("context", "")
            question = row.get("question", "")
            answer_letter = row.get("answer", "")
            if not context or not question:
                continue

            if len(context) > max_context_chars:
                context = context[:max_context_chars]

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


def download_narrativeqa_direct(max_n=0, max_context_chars=260000, cache_name=None):
    """NarrativeQA from deepmind — streaming mode, truncate long stories.

    If cache_name is provided, writes directly to cache (constant memory)
    and returns None. Otherwise returns a list of samples.
    """
    from datasets import load_dataset

    writer = _StreamingCacheWriter(cache_name) if cache_name else None
    samples = [] if writer is None else None
    total_count = 0
    try:
        print("    Loading deepmind/narrativeqa (streaming)...", flush=True)
        for split in ["train", "validation"]:
            ds = load_dataset("deepmind/narrativeqa", split=split,
                              token=HF_TOKEN, streaming=True)

            count = 0
            for row in _iter_progress(ds, desc=f"narrativeqa-{split}",
                                      check_memory_ctx="narrativeqa_direct"):
                doc = row.get("document", {})
                text = doc.get("text", "").strip()
                if not text:
                    summary = doc.get("summary", "")
                    if isinstance(summary, dict):
                        text = summary.get("text", "").strip()
                    elif isinstance(summary, str):
                        text = summary.strip()
                if not text:
                    continue

                # Truncate to cap memory
                if len(text) > max_context_chars:
                    text = text[:max_context_chars]

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

                sample = {
                    "context": text,
                    "question": question,
                    "answer": answer,
                    "source": f"narrativeqa_{split}",
                }
                if writer:
                    writer.write(sample)
                else:
                    samples.append(sample)

                count += 1
                if max_n and count >= max_n:
                    break
            total_count += count
            print(f"      {split}: {count} samples", flush=True)
    except Exception as e:
        print(f"      FAILED: {e}", flush=True)
        traceback.print_exc()

    if writer:
        writer.close()
        print(f"  Total from narrativeqa_direct: {total_count:,}", flush=True)
        return None
    return samples


def download_scrolls_narrativeqa(max_n=0, max_context_chars=260000, cache_name=None):
    """NarrativeQA via SCROLLS zip — stream line-by-line, write directly to cache.

    If cache_name is provided, writes directly to cache (constant memory)
    and returns None. Otherwise returns a list of samples.
    """
    print(f"    Downloading tau/scrolls narrative_qa.zip ...", flush=True)
    try:
        zip_path = _hf_download("tau/scrolls", "narrative_qa.zip")
    except Exception as e:
        print(f"      DOWNLOAD FAILED: {e}", flush=True)
        traceback.print_exc()
        return [] if cache_name is None else None

    zip_size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"    Zip downloaded: {zip_size_mb:.1f} MB  (RSS={_get_current_rss_gb():.2f}GB)", flush=True)

    writer = _StreamingCacheWriter(cache_name) if cache_name else None
    samples = [] if writer is None else None
    total_count = 0

    for split in ["train", "validation"]:
        try:
            inner = f"narrative_qa/{split}.jsonl"
            print(f"    Streaming tau/scrolls narrative_qa/{split}...", flush=True)

            count = 0
            with zipfile.ZipFile(zip_path, "r") as zf:
                # Check that the inner file exists
                if inner not in zf.namelist():
                    print(f"      ERROR: '{inner}' not found in zip. Available: {zf.namelist()[:10]}", flush=True)
                    continue

                inner_info = zf.getinfo(inner)
                print(f"      {inner}: compressed={inner_info.compress_size/(1024*1024):.1f}MB "
                      f"uncompressed={inner_info.file_size/(1024*1024):.1f}MB", flush=True)

                # Max bytes per JSON line: skip pathologically large lines
                # (a full novel as JSON ≈ 1-5 MB; anything > 20 MB is likely corrupt)
                MAX_LINE_BYTES = 20 * 1024 * 1024  # 20 MB

                with zf.open(inner) as f:
                    for line_bytes in _iter_progress(f, desc=f"scrolls-nqa-{split}",
                                                     check_memory_ctx="scrolls_narrativeqa"):
                        # Guard: skip overly large lines to prevent memory spikes
                        if len(line_bytes) > MAX_LINE_BYTES:
                            print(f"      WARNING: skipping oversized line ({len(line_bytes)/(1024*1024):.1f}MB)", flush=True)
                            del line_bytes
                            continue

                        line = line_bytes.decode("utf-8").strip()
                        del line_bytes  # free raw bytes immediately
                        if not line:
                            continue
                        try:
                            row = json.loads(line)
                        except json.JSONDecodeError as je:
                            print(f"      WARNING: bad JSON line (len={len(line)}): {je}", flush=True)
                            del line
                            continue
                        del line  # free decoded string; data now in row dict

                        inp = row.get("input", "")
                        output = row.get("output", "")
                        del row  # free parsed JSON immediately
                        if not inp or not output:
                            del inp, output
                            continue

                        # SCROLLS: question first, then document
                        # Use find() to avoid creating intermediate list from split()
                        sep_pos = inp.find("\n\n")
                        if sep_pos != -1 and sep_pos < 500:
                            question = inp[:sep_pos].strip()
                            context = inp[sep_pos + 2:].strip()
                        else:
                            sep_pos = inp.find("\n")
                            if sep_pos != -1:
                                question = inp[:sep_pos].strip()
                                context = inp[sep_pos + 1:].strip()
                            else:
                                del inp
                                continue

                        # Free the large input string immediately
                        del inp

                        if not context or not question or len(context) < 200:
                            del context, question
                            continue

                        # Truncate to cap memory
                        if len(context) > max_context_chars:
                            context = context[:max_context_chars]

                        sample = {
                            "context": context,
                            "question": question,
                            "answer": output.strip(),
                            "source": f"scrolls_narrativeqa_{split}",
                        }
                        del context, question, output
                        if writer:
                            writer.write(sample)
                        else:
                            samples.append(sample)
                        del sample

                        count += 1
                        # Periodic gc to prevent memory fragmentation
                        if count % 2000 == 0:
                            gc.collect()
                        if max_n and count >= max_n:
                            break
            total_count += count
            print(f"      {split}: {count} samples  (RSS={_get_current_rss_gb():.2f}GB)", flush=True)
            gc.collect()
        except MemoryError as e:
            print(f"\n      MEMORY ERROR ({split}): {e}", flush=True, file=sys.stderr)
            if writer:
                writer.close()
            raise
        except Exception as e:
            print(f"      FAILED ({split}): {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()

    if writer:
        writer.close()
        print(f"  Total from scrolls_narrativeqa: {total_count:,}", flush=True)
        return None
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

# Large datasets: stream directly to cache (constant memory during download)
_STREAMING_DATASETS = {
    "narrativeqa_direct":  download_narrativeqa_direct,
    "scrolls_narrativeqa": download_scrolls_narrativeqa,
}

# Functions that accept max_context_chars (non-streaming)
_DATASETS_WITH_TRUNCATE = {
    "longbench_v2":        download_longbench_v2,
}

# Functions that don't need it (contexts already reasonable)
_DATASETS_PLAIN = {
    "longbench":       download_longbench,
    "quality_direct":  download_quality_direct,
    "scrolls_quality": download_scrolls_quality,
    "scrolls_qasper":  download_scrolls_qasper,
    "leval":           download_leval,
}

# Ordered download list
DATASET_NAMES = [
    "longbench", "longbench_v2", "narrativeqa_direct", "scrolls_narrativeqa",
    "quality_direct", "scrolls_quality", "scrolls_qasper", "leval",
]

# Datasets that are redundant with each other (same underlying novels/docs).
# Key = dataset that can be skipped, value = dataset it duplicates.
_REDUNDANT_PAIRS = {
    "scrolls_narrativeqa": "narrativeqa_direct",
}


def _process_chunk(chunk, tokenizer, args, do_truncate, out_f,
                   all_lengths_out, source_counts_out):
    """Tokenize a chunk of samples, filter by token length, write survivors.

    Modifies all_lengths_out and source_counts_out in place.
    """
    # Tokenize
    lengths = [len(tokenizer(s["context"], return_tensors=None, padding=False,
                             truncation=False)["input_ids"]) for s in chunk]

    for s, tok_len in zip(chunk, lengths):
        # Truncate if needed
        if do_truncate and tok_len > args.max_tokens:
            ratio = args.max_tokens / tok_len
            max_chars = int(len(s["context"]) * ratio * 0.95)
            s["context"] = s["context"][:max_chars]
            # Re-tokenize truncated doc
            tok_len = len(tokenizer(s["context"], return_tensors=None, padding=False,
                                    truncation=False)["input_ids"])

        # Filter by range
        if args.min_tokens <= tok_len <= args.max_tokens:
            out_f.write(json.dumps(s, ensure_ascii=False) + "\n")
            all_lengths_out.append(tok_len)
            source_counts_out[s.get("source", "unknown")] += 1


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
    parser.add_argument("--max_context_chars", type=int, default=260000,
                        help="Truncate context during download (chars, default: 260000 ≈ 64K tokens)")
    parser.add_argument("--model", default="models/Qwen3-0.6B",
                        help="Tokenizer path (default: models/Qwen3-0.6B)")
    parser.add_argument("--num_workers", type=int, default=4)
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
    parser.add_argument("--max_memory_gb", type=float, default=0,
                        help="Abort gracefully when RSS exceeds this (GB). "
                             "0 = no limit (default). E.g. --max_memory_gb 6")
    args = parser.parse_args()

    global HF_TOKEN, _MEMORY_LIMIT_GB
    if args.hf_token:
        HF_TOKEN = args.hf_token
    if args.max_memory_gb > 0:
        _MEMORY_LIMIT_GB = args.max_memory_gb
        print(f"  Memory limit: {_MEMORY_LIMIT_GB:.1f} GB", flush=True)
    do_truncate = not args.no_truncate

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    max_n = 50 if args.test else 0

    # ── Download ──
    print("\n" + "=" * 70)
    print("  DOWNLOADING LONG-DOCUMENT QA DATASETS")
    print("=" * 70)

    downloaded_names = []
    skip = set(args.skip)
    only = set(args.only) if args.only else None
    mcc = args.max_context_chars

    for name in DATASET_NAMES:
        if name in skip:
            print(f"\n  SKIP: {name}", flush=True); continue
        if only and name not in only:
            print(f"\n  SKIP: {name} (not in --only)", flush=True); continue

        # Warn about redundant datasets but still process them (different questions)
        if name in _REDUNDANT_PAIRS:
            primary = _REDUNDANT_PAIRS[name]
            if _cache_exists(primary):
                primary_count = _cache_count(primary)
                print(f"\n  NOTE: {name} shares documents with {primary} ({primary_count:,} samples),")
                print(f"        but has different questions — keeping both.", flush=True)
                print(f"        (use --skip {name} to exclude)", flush=True)

        print(f"\n  ── {name} ──", flush=True)
        if not getattr(args, 'no_cache', False) and _cache_exists(name):
            count = _cache_count(name)
            print(f"  CACHED: {count:,} samples", flush=True)
            downloaded_names.append(name)
            continue

        with _Timer(name):
            try:
                if name in _STREAMING_DATASETS:
                    # Large datasets: stream directly to cache (constant memory)
                    fn = _STREAMING_DATASETS[name]
                    fn(max_n=max_n, max_context_chars=mcc, cache_name=name)
                elif name in _DATASETS_WITH_TRUNCATE:
                    samples = _DATASETS_WITH_TRUNCATE[name](max_n=max_n, max_context_chars=mcc)
                    if samples:
                        _cache_save(name, samples)
                    print(f"  Total from {name}: {len(samples):,}", flush=True)
                    del samples
                else:
                    samples = _DATASETS_PLAIN[name](max_n=max_n)
                    if samples:
                        _cache_save(name, samples)
                    print(f"  Total from {name}: {len(samples):,}", flush=True)
                    del samples
            except MemoryError as e:
                print(f"\n  MEMORY ERROR during {name}: {e}", flush=True, file=sys.stderr)
                print(f"  RSS at failure: {_get_current_rss_gb():.2f} GB", flush=True, file=sys.stderr)
                traceback.print_exc()
                # Save partial progress and continue with other datasets
                gc.collect()
                continue
            except Exception as e:
                print(f"  ERROR during {name}: {type(e).__name__}: {e}", flush=True)
                print(f"  RSS at failure: {_get_current_rss_gb():.2f} GB", flush=True)
                traceback.print_exc()
                continue

        if _cache_exists(name):
            downloaded_names.append(name)
        gc.collect()

    # ── Streaming merge: clean + dedup (two-pass, constant memory) ──
    #
    # Problem: loading all caches into memory at once uses 16+ GB.
    # Solution:
    #   Pass 1: stream all caches, collect dedup keys (16 bytes each ≈ 1.3 MB for 80K samples)
    #   Pass 2: stream again, skip dupes + bad samples, write to cleaned JSONL on disk
    #
    print("\n" + "=" * 70)
    print("  STREAMING CLEAN + DEDUP")
    print("=" * 70)

    # Summarize cache sizes
    total_cache_mb = 0
    for name in downloaded_names:
        sz = _cache_file_size_mb(name)
        total_cache_mb += sz
        print(f"    {name}: {_cache_count(name):,} samples ({sz:.1f} MB on disk)", flush=True)
    print(f"    Total on disk: {total_cache_mb:.0f} MB", flush=True)
    print(f"    RSS before merge: {_get_current_rss_gb():.2f} GB", flush=True)

    # Pass 1: build dedup key set (stream, constant memory except for the set itself)
    print(f"\n  Pass 1/2: collecting dedup keys...", flush=True)
    dedup_keys = set()
    raw_count = 0
    dup_count = 0
    for name in downloaded_names:
        for s in _cache_iter(name):
            raw_count += 1
            ctx_prefix = (s.get("context") or "")[:500]
            q = s.get("question") or ""
            key = hashlib.md5((ctx_prefix + "||" + q).encode()).hexdigest()
            if key in dedup_keys:
                dup_count += 1
            else:
                dedup_keys.add(key)
    print(f"    Raw: {raw_count:,}  Unique keys: {len(dedup_keys):,}  Dups: {dup_count:,}", flush=True)

    # Pass 2: stream again, apply cleaning + dedup, write to combined JSONL
    print(f"\n  Pass 2/2: cleaning + writing...", flush=True)
    combined_path = DATA_DIR / ".longdoc_combined_clean.jsonl"
    seen_keys = set()
    reasons = Counter()
    clean_count = 0
    with open(combined_path, "w", encoding="utf-8") as out_f:
        for name in downloaded_names:
            name_count = 0
            for s in _cache_iter(name):
                # Cleaning (inline, same logic as clean_data)
                ctx = (s.get("context") or "").strip()
                q = (s.get("question") or "").strip()
                a = (s.get("answer") or "").strip()
                if not ctx: reasons["missing_context"] += 1; continue
                if not q: reasons["missing_question"] += 1; continue
                if not a: reasons["missing_answer"] += 1; continue
                if len(ctx) < 50: reasons["short_context"] += 1; continue
                if len(q) < 2: reasons["short_question"] += 1; continue
                if _normalize_text(q) == _normalize_text(a): reasons["q_equals_a"] += 1; continue

                # Dedup
                key = hashlib.md5((ctx[:500] + "||" + q).encode()).hexdigest()
                if key in seen_keys:
                    continue
                seen_keys.add(key)

                s["context"] = ctx; s["question"] = q; s["answer"] = a
                out_f.write(json.dumps(s, ensure_ascii=False) + "\n")
                clean_count += 1
                name_count += 1
            gc.collect()
            print(f"    {name}: {name_count:,} kept  (RSS={_get_current_rss_gb():.2f}GB)", flush=True)

    del dedup_keys, seen_keys
    gc.collect()

    removed = raw_count - clean_count
    print(f"\n  Cleaning: {raw_count:,} -> {clean_count:,} ({removed:,} removed)")
    for r, c in reasons.most_common():
        print(f"    - {r}: {c:,}")
    print(f"  RSS after clean+dedup: {_get_current_rss_gb():.2f} GB", flush=True)

    if clean_count == 0:
        print("\n  ERROR: No samples after cleaning.")
        sys.exit(1)

    # ── Token lengths & filter (chunked to limit memory) ──
    print("\n" + "=" * 70)
    print(f"  TOKEN FILTERING [{args.min_tokens:,} – {args.max_tokens:,}]"
          f"  truncate={'ON' if do_truncate else 'OFF'}")
    print("=" * 70)

    CHUNK_SIZE = 2000
    train_path = DATA_DIR / f"{args.output_prefix}_train.json"
    dev_path = DATA_DIR / f"{args.output_prefix}_dev.json"

    # Chunked tokenization + filtering: read combined JSONL in chunks,
    # tokenize, filter, write survivors to temp file with their lengths.
    filtered_path = DATA_DIR / ".longdoc_filtered.jsonl"
    total_before_filter = 0
    total_after_filter = 0
    all_lengths_for_stats = []
    source_counts = Counter()

    from transformers import AutoTokenizer
    print(f"    Loading tokenizer from {args.model}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    print(f"    Tokenizing & filtering in chunks of {CHUNK_SIZE:,}...", flush=True)
    t0 = time.time()
    chunk_buf = []
    with open(filtered_path, "w", encoding="utf-8") as out_f:
        with open(combined_path, "r", encoding="utf-8") as in_f:
            for line in in_f:
                line = line.strip()
                if not line:
                    continue
                chunk_buf.append(json.loads(line))
                if len(chunk_buf) >= CHUNK_SIZE:
                    _process_chunk(chunk_buf, tokenizer, args, do_truncate, out_f,
                                   all_lengths_for_stats, source_counts)
                    total_before_filter += len(chunk_buf)
                    chunk_buf = []

            # Process remaining
            if chunk_buf:
                _process_chunk(chunk_buf, tokenizer, args, do_truncate, out_f,
                               all_lengths_for_stats, source_counts)
                total_before_filter += len(chunk_buf)
                chunk_buf = []

    total_after_filter = len(all_lengths_for_stats)
    elapsed = time.time() - t0
    print(f"    Tokenized {total_before_filter:,} samples in {elapsed:.1f}s", flush=True)
    print(f"    Filtered: {total_before_filter:,} -> {total_after_filter:,} "
          f"({total_before_filter - total_after_filter:,} dropped)", flush=True)
    print(f"    RSS after tokenization: {_get_current_rss_gb():.2f} GB", flush=True)

    del tokenizer
    gc.collect()

    if total_after_filter == 0:
        print(f"\n  WARNING: 0 samples in [{args.min_tokens}, {args.max_tokens}]!")
        print(f"  Falling back to all {total_before_filter:,} samples...")
        # Copy combined -> filtered without token filter
        import shutil
        shutil.copy2(combined_path, filtered_path)
        total_after_filter = total_before_filter

    # Print stats
    if all_lengths_for_stats:
        arr = np.array(all_lengths_for_stats)
        p = np.percentile(arr, [10, 25, 50, 75, 90, 95, 99])
        print(f"\n{'=' * 70}")
        print(f"  TOKEN LENGTH STATS [AFTER filter] ({len(arr):,} samples)")
        print(f"{'=' * 70}")
        print(f"  Min:  {arr.min():>8,}    Max: {arr.max():>8,}    Mean: {arr.mean():>10.1f}")
        print(f"  P10:  {int(p[0]):>8,}    P25: {int(p[1]):>8,}    P50:  {int(p[2]):>8,}")
        print(f"  P75:  {int(p[3]):>8,}    P90: {int(p[4]):>8,}    P95:  {int(p[5]):>8,}")
        print(f"  P99:  {int(p[6]):>8,}")

        buckets = [0, 1024, 2048, 4096, 8192, 16384, 32768, 65536, float("inf")]
        blabels = ["<1K", "1-2K", "2-4K", "4-8K", "8-16K", "16-32K", "32-64K", ">64K"]
        print(f"\n  Distribution:")
        for i in range(len(blabels)):
            cnt = int(np.sum((arr >= buckets[i]) & (arr < buckets[i + 1])))
            pct = 100.0 * cnt / len(arr)
            bar = "█" * int(40 * cnt / max(len(arr), 1))
            print(f"    {blabels[i]:>8} │{bar:<40} {cnt:>8,} ({pct:5.1f}%)")

        print(f"\n  Per-source:")
        for src, cnt in source_counts.most_common():
            print(f"    {src:<35} {cnt:>6,}")

    del all_lengths_for_stats
    gc.collect()

    # ── Split & Save (stream from filtered JSONL) ──
    print("\n" + "=" * 70)
    print("  SPLIT TRAIN / DEV & SAVE")
    print("=" * 70)

    rng = random.Random(args.seed)
    # Determine dev indices: pick dev_ratio of [0..total_after_filter) as dev
    indices = list(range(total_after_filter))
    rng.shuffle(indices)
    dev_size = max(1, int(total_after_filter * args.dev_ratio))
    dev_set = set(indices[:dev_size])
    del indices

    train_count = 0
    dev_count = 0
    with open(train_path, "w", encoding="utf-8") as tf, \
         open(dev_path, "w", encoding="utf-8") as df:
        tf.write("[\n")
        df.write("[\n")
        idx = 0
        with open(filtered_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if idx in dev_set:
                    if dev_count > 0:
                        df.write(",\n")
                    df.write(line)
                    dev_count += 1
                else:
                    if train_count > 0:
                        tf.write(",\n")
                    tf.write(line)
                    train_count += 1
                idx += 1
        tf.write("\n]")
        df.write("\n]")

    train_mb = train_path.stat().st_size / (1024 * 1024)
    dev_mb = dev_path.stat().st_size / (1024 * 1024)
    print(f"  Train: {train_path.name}  ({train_count:,} samples, {train_mb:.1f} MB)")
    print(f"  Dev:   {dev_path.name}  ({dev_count:,} samples, {dev_mb:.1f} MB)")

    # Clean up temp files
    combined_path.unlink(missing_ok=True)
    filtered_path.unlink(missing_ok=True)

    # ── Summary ──
    print("\n" + "=" * 70)
    print("  DONE")
    print("=" * 70)
    print(f"  Token range: [{args.min_tokens:,}, {args.max_tokens:,}]")
    print(f"  Train: {train_count:,}    Dev: {dev_count:,}")
    print(f"  Sources:")
    for src, cnt in source_counts.most_common():
        print(f"    {src:<35} {cnt:>6,}")
    print(f"\n  {train_path}\n  {dev_path}")
    print(f"  Peak RSS: {_get_rss_gb():.2f} GB")
    print("=" * 70)


if __name__ == "__main__":
    main()
