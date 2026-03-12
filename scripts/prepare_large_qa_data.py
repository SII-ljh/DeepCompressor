#!/usr/bin/env python3
"""Prepare large-scale QA datasets for pure QA training (without Stage 1 NTP).

Downloads QA data from 16+ sources, performs data cleaning, and saves in
standard format compatible with the deep_compressor QA pipeline.

Output format (JSON array, saved as qa_large_train.json / qa_large_dev.json):
  [{"context": str, "question": str, "answer": str, "source": str}, ...]

Data cleaning removes:
  - Samples missing context, question, or answer
  - Samples where question == answer (exact match after normalization)
  - Unanswerable / "No answer" samples
  - Samples with very short fields (context < 50 chars, question < 2 chars)

Final statistics report:
  - Per-source sample counts and cleaning breakdown
  - Duplicate question / answer analysis

Datasets (16):
  English:  SQuAD v1, SQuAD v2, TriviaQA, Natural Questions, HotpotQA,
            QuAC, DROP, AdversarialQA, DuoRC(SelfRC)
  Chinese:  CMRC2018, DuReader, DRCD, WebQA
  Multi:    MLQA (en+zh), XQuAD (en+zh)

Usage:
    python scripts/prepare_large_qa_data.py                  # full download
    python scripts/prepare_large_qa_data.py --test           # small subset
    python scripts/prepare_large_qa_data.py --only-chinese   # Chinese only
    python scripts/prepare_large_qa_data.py --only-english   # English only
"""

import argparse
import json
import random
import sys
import time
import traceback
import unicodedata
from collections import Counter
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# ========== Text normalization for dedup / cleaning ==========

def _normalize_text(text: str) -> str:
    """Normalize text for comparison: lowercase, strip, collapse whitespace."""
    text = text.strip().lower()
    text = unicodedata.normalize("NFKC", text)
    text = " ".join(text.split())
    return text


# ========== Progress helpers ==========

def _iter_with_progress(iterable, desc="Processing", every=10000):
    """Wrap an iterable to print progress every `every` items.

    Crucial for streaming datasets where download happens during iteration.
    """
    count = 0
    t0 = time.time()
    last_print = t0
    for item in iterable:
        count += 1
        now = time.time()
        # Print every N items OR every 30 seconds (whichever comes first)
        if count % every == 0 or (now - last_print) >= 30:
            elapsed = now - t0
            rate = count / max(elapsed, 0.01)
            print(f"    ... {desc}: {count:,} rows downloaded ({elapsed:.0f}s, {rate:.0f} rows/s)",
                  flush=True)
            last_print = now
        yield item
    elapsed = time.time() - t0
    rate = count / max(elapsed, 0.01)
    print(f"    ... {desc}: {count:,} rows total ({elapsed:.0f}s, {rate:.0f} rows/s)", flush=True)


class _Timer:
    """Simple context manager to track and print elapsed time."""
    def __init__(self, label):
        self.label = label
        self.t0 = None

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *args):
        elapsed = time.time() - self.t0
        print(f"  [{self.label}] elapsed: {elapsed:.1f}s", flush=True)


# ========== Dataset cache for resume support ==========

CACHE_DIR = DATA_DIR / ".qa_cache"


def _cache_path(name: str) -> Path:
    return CACHE_DIR / f"{name}.json"


def _cache_exists(name: str) -> bool:
    return _cache_path(name).exists()


def _cache_save(name: str, train: list, dev: list):
    """Save dataset results to cache so we can skip re-downloading."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(_cache_path(name), "w", encoding="utf-8") as f:
        json.dump({"train": train, "dev": dev}, f, ensure_ascii=False)
    print(f"    -> cached to {_cache_path(name).name} "
          f"(train={len(train):,}, dev={len(dev):,})", flush=True)


def _cache_load(name: str) -> tuple:
    """Load cached dataset. Returns (train_list, dev_list)."""
    with open(_cache_path(name), "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["train"], data["dev"]


def _try_load_cache(name: str, train_all: list, dev_all: list) -> bool:
    """If cache exists, load it into train_all/dev_all and return True."""
    if not _cache_exists(name):
        return False
    cached_train, cached_dev = _cache_load(name)
    train_all.extend(cached_train)
    dev_all.extend(cached_dev)
    print(f"  CACHED: train={len(cached_train):,}, dev={len(cached_dev):,} "
          f"(from {_cache_path(name).name})", flush=True)
    return True


def clear_cache():
    """Delete all cached dataset files."""
    if CACHE_DIR.exists():
        import shutil
        shutil.rmtree(CACHE_DIR)
        print(f"Cache cleared: {CACHE_DIR}", flush=True)
    else:
        print("No cache to clear.", flush=True)


# ========== Conversion functions ==========

def _convert_squad(dataset, max_n: int = 0, source: str = "squad",
                   include_unanswerable: bool = False):
    """Convert SQuAD-format dataset (also works for AdversarialQA, MLQA, XQuAD)."""
    samples = []
    for row in dataset:
        if max_n and len(samples) >= max_n:
            break
        answers = row.get("answers", {}).get("text", [])
        if not answers:
            if include_unanswerable:
                samples.append({
                    "context": row["context"],
                    "question": row["question"],
                    "answer": "无法回答" if "chinese" in source.lower() else "No answer",
                    "source": source,
                    "unanswerable": True,
                })
            continue
        samples.append({
            "context": row["context"],
            "question": row["question"],
            "answer": answers[0],
            "source": source,
        })
    return samples


def _convert_cmrc_or_drcd(dataset, max_n: int = 0, source: str = "cmrc"):
    """Convert CMRC2018 / DRCD dataset."""
    samples = []
    for row in dataset:
        if max_n and len(samples) >= max_n:
            break
        answers = row.get("answers", {}).get("text", [])
        if not answers:
            continue
        samples.append({
            "context": row["context"],
            "question": row["question"],
            "answer": answers[0],
            "source": source,
        })
    return samples


def _convert_dureader(dataset, max_n: int = 0, source: str = "dureader"):
    """Convert DuReader-robust dataset.

    Supports two formats:
    1. Flat format: {"context": ..., "question": ..., "answers": {"text": [...]}}
    2. Nested SQuAD format (dirtycomputer): {"data": {"paragraphs": [{"context": ..., "qas": [...]}]}}
    """
    samples = []
    for row in dataset:
        if max_n and len(samples) >= max_n:
            break

        # Nested SQuAD format (dirtycomputer/dureader_robust-data)
        if "data" in row and isinstance(row["data"], dict):
            paragraphs = row["data"].get("paragraphs", [])
            for para in paragraphs:
                if max_n and len(samples) >= max_n:
                    break
                context = para.get("context", "")
                if not context:
                    continue
                for qa in para.get("qas", []):
                    if max_n and len(samples) >= max_n:
                        break
                    question = qa.get("question", "")
                    answers = qa.get("answers", [])
                    if not answers or not question:
                        continue
                    answer = answers[0].get("text", "") if isinstance(answers[0], dict) else str(answers[0])
                    if not answer:
                        continue
                    samples.append({
                        "context": context,
                        "question": question,
                        "answer": answer,
                        "source": source,
                    })
            continue

        # Flat format
        answers = row.get("answers", {}).get("text", [])
        if not answers:
            answer = row.get("answer", "")
            if not answer:
                continue
            answers = [answer]
        samples.append({
            "context": row["context"],
            "question": row["question"],
            "answer": answers[0],
            "source": source,
        })
    return samples


def _convert_triviaqa(dataset, max_n: int = 0, max_context_chars: int = 32000,
                      source: str = "triviaqa"):
    """Convert TriviaQA RC dataset."""
    samples = []
    for row in dataset:
        if max_n and len(samples) >= max_n:
            break
        answer_val = row.get("answer", {}).get("value", "")
        if not answer_val:
            continue

        # Extract context
        context = ""
        wiki_ctxs = row.get("entity_pages", {}).get("wiki_context", [])
        if wiki_ctxs:
            context = wiki_ctxs[0].strip()
        if not context or len(context) < 100:
            search_ctxs = row.get("search_results", {}).get("search_context", [])
            if search_ctxs:
                context = search_ctxs[0].strip()
        if not context or len(context) < 100:
            continue

        if len(context) > max_context_chars:
            context = context[:max_context_chars]

        samples.append({
            "context": context,
            "question": row["question"],
            "answer": answer_val,
            "source": source,
        })
    return samples


def _convert_natural_questions(dataset, max_n: int = 0, max_context_chars: int = 32000,
                               source: str = "natural_questions"):
    """Convert Natural Questions dataset.

    Only keeps samples with short answers (ignores long-answer-only and no-answer).

    HuggingFace NQ format (per row):
      annotations.short_answers: list of 5 annotator dicts, each with:
        text: list[str]        — answer text spans (empty list = no answer)
        start_token: list[int] — start token indices
        end_token:   list[int] — end token indices
      document.tokens: {"token": list[str], "is_html": list[bool]}
      question: {"text": str, "tokens": list[str]}
    """
    samples = []
    for row in dataset:
        if max_n and len(samples) >= max_n:
            break

        # Find the first annotator with a non-empty short answer
        sa_list = row.get("annotations", {}).get("short_answers", [])
        answer = ""
        for sa in sa_list:
            texts = sa.get("text", [])
            if texts and len(texts) > 0 and texts[0]:
                answer = texts[0]
                break

        if not answer:
            continue

        # Build context from document tokens (document.text is usually empty)
        doc_tokens_raw = row.get("document", {}).get("tokens", {})
        if isinstance(doc_tokens_raw, dict):
            token_list = doc_tokens_raw.get("token", [])
            is_html = doc_tokens_raw.get("is_html", [])
        else:
            token_list = []
            is_html = []

        context = row.get("document", {}).get("text", "")
        if not context and token_list:
            # Build context from non-HTML tokens
            if is_html:
                context = " ".join(t for t, h in zip(token_list, is_html) if not h)
            else:
                context = " ".join(token_list)

        if not context or len(context) < 100:
            continue

        if len(context) > max_context_chars:
            context = context[:max_context_chars]

        # question is {"text": str} or plain string
        question = row.get("question", "")
        if isinstance(question, dict):
            question = question.get("text", "")

        samples.append({
            "context": context,
            "question": question,
            "answer": answer.strip(),
            "source": source,
        })

    return samples


def _convert_hotpotqa(dataset, max_n: int = 0, max_context_chars: int = 32000,
                      source: str = "hotpotqa"):
    """Convert HotpotQA dataset (multi-hop reasoning)."""
    samples = []
    for row in dataset:
        if max_n and len(samples) >= max_n:
            break

        answer = row.get("answer", "").strip()
        if not answer:
            continue

        contexts = row.get("context", {}).get("sentences", [])
        if not contexts:
            titles = row.get("context", {}).get("title", [])
            contexts = titles

        if not contexts:
            continue

        if isinstance(contexts[0], list):
            context = " ".join(" ".join(sents) for sents in contexts)
        else:
            context = " ".join(contexts)

        if len(context) < 100:
            continue

        if len(context) > max_context_chars:
            context = context[:max_context_chars]

        samples.append({
            "context": context,
            "question": row["question"],
            "answer": answer,
            "source": source,
        })

    return samples


def _convert_webqa(dataset, max_n: int = 0, max_context_chars: int = 32000,
                   source: str = "webqa"):
    """Convert WebQA (Chinese web QA) dataset."""
    samples = []
    for row in dataset:
        if max_n and len(samples) >= max_n:
            break

        answers = row.get("answers", [])
        if not answers:
            answer = row.get("answer", "").strip()
            if not answer:
                continue
        else:
            answer = answers[0] if isinstance(answers, list) else answers

        context = row.get("evidence", "")
        if not context:
            context = row.get("context", "")

        if not context or len(context) < 50:
            continue

        if len(context) > max_context_chars:
            context = context[:max_context_chars]

        samples.append({
            "context": context,
            "question": row.get("question", ""),
            "answer": answer,
            "source": source,
        })

    return samples


# ========== NEW conversion functions ==========

def _convert_quac(dataset, max_n: int = 0, source: str = "quac"):
    """Convert QuAC conversational QA dataset.

    Each row is a dialog with multiple Q/A turns over the same context.
    We flatten into independent (context, question, answer) triples.
    Skips CANNOTANSWER turns.
    """
    samples = []
    for row in dataset:
        if max_n and len(samples) >= max_n:
            break

        context = row.get("context", "")
        if not context:
            # QuAC also stores background info
            background = row.get("background", "")
            section_title = row.get("section_title", "")
            context = f"{section_title}\n{background}" if section_title else background

        if not context or len(context) < 50:
            continue

        questions = row.get("questions", [])
        answers_dict = row.get("answers", {})
        answer_texts = answers_dict.get("input_text", [])

        for q, a in zip(questions, answer_texts):
            if max_n and len(samples) >= max_n:
                break
            if not a or not q:
                continue
            # QuAC uses "CANNOTANSWER" for unanswerable
            if a.strip() == "CANNOTANSWER":
                continue
            samples.append({
                "context": context,
                "question": q,
                "answer": a.strip(),
                "source": source,
            })

    return samples


def _convert_drop(dataset, max_n: int = 0, source: str = "drop"):
    """Convert DROP dataset (discrete reasoning over paragraphs)."""
    samples = []
    for row in dataset:
        if max_n and len(samples) >= max_n:
            break

        passage = row.get("passage", "")
        question = row.get("question", "")
        if not passage or not question:
            continue

        # DROP answers: try spans first, then number, then date
        answers_spans = row.get("answers_spans", {})
        spans = answers_spans.get("spans", [])
        if spans and spans[0]:
            answer = spans[0]
        else:
            # Fallback: try answer field directly
            answer = row.get("answer", "")
            if not answer:
                continue

        if not answer.strip():
            continue

        samples.append({
            "context": passage,
            "question": question,
            "answer": answer.strip(),
            "source": source,
        })

    return samples


def _convert_duorc(dataset, max_n: int = 0, source: str = "duorc"):
    """Convert DuoRC dataset (movie plot QA)."""
    samples = []
    for row in dataset:
        if max_n and len(samples) >= max_n:
            break

        context = row.get("plot", "")
        question = row.get("question", "")
        answers = row.get("answers", [])

        if not context or not question or not answers:
            continue

        # Skip no_answer samples
        if row.get("no_answer", False):
            continue

        # answers is a list of strings
        answer = answers[0]
        if isinstance(answer, dict):
            answer = answer.get("text", "")
        answer = str(answer).strip()

        if not answer or answer.lower() == "no answer":
            continue

        samples.append({
            "context": context,
            "question": question,
            "answer": answer,
            "source": source,
        })

    return samples


def _convert_coqa(dataset, max_n: int = 0, source: str = "coqa"):
    """Convert CoQA conversational QA dataset.

    Each row is a story with multiple Q/A turns. Flatten to independent samples.
    """
    samples = []
    for row in dataset:
        if max_n and len(samples) >= max_n:
            break

        story = row.get("story", "")
        if not story or len(story) < 50:
            continue

        questions = row.get("questions", [])
        answers_dict = row.get("answers", {})
        answer_texts = answers_dict.get("input_text", [])

        for q, a in zip(questions, answer_texts):
            if max_n and len(samples) >= max_n:
                break
            if not q or not a:
                continue
            if a.strip().lower() in ("unknown", "no answer"):
                continue
            samples.append({
                "context": story,
                "question": q,
                "answer": a.strip(),
                "source": source,
            })

    return samples


# ========== Data cleaning ==========

def clean_data(samples: list, label: str = "data") -> list:
    """Clean QA samples and report statistics.

    Removes:
      1. Missing fields (context, question, answer)
      2. Very short fields (context < 50 chars, question < 2 chars)
      3. Unanswerable samples (unanswerable=True or answer is "No answer"/"无法回答")
      4. question == answer (exact match after normalization)

    Returns cleaned list.
    """
    total = len(samples)
    reasons = Counter()
    cleaned = []

    no_answer_markers = {
        "no answer", "无法回答", "unanswerable", "cannotanswer", "unknown",
        "no_answer", "none", "n/a",
    }

    for s in samples:
        ctx = (s.get("context") or "").strip()
        q = (s.get("question") or "").strip()
        a = (s.get("answer") or "").strip()

        # 1. Missing fields
        if not ctx:
            reasons["missing_context"] += 1
            continue
        if not q:
            reasons["missing_question"] += 1
            continue
        if not a:
            reasons["missing_answer"] += 1
            continue

        # 2. Short fields
        if len(ctx) < 50:
            reasons["short_context"] += 1
            continue
        if len(q) < 2:
            reasons["short_question"] += 1
            continue

        # 3. Unanswerable
        if s.get("unanswerable", False):
            reasons["unanswerable_flag"] += 1
            continue
        if _normalize_text(a) in no_answer_markers:
            reasons["unanswerable_text"] += 1
            continue

        # 4. Q == A
        if _normalize_text(q) == _normalize_text(a):
            reasons["q_equals_a"] += 1
            continue

        # Clean fields in place
        s["context"] = ctx
        s["question"] = q
        s["answer"] = a
        # Remove unanswerable key if present (keep format clean)
        s.pop("unanswerable", None)

        cleaned.append(s)

    removed = total - len(cleaned)
    print(f"\n  [{label}] Cleaning: {total:,} -> {len(cleaned):,} ({removed:,} removed, {removed/max(total,1)*100:.1f}%)")
    if reasons:
        for reason, count in reasons.most_common():
            print(f"    - {reason}: {count:,}")

    return cleaned


# ========== Duplicate analysis ==========

def report_duplicates(samples: list, label: str = "data"):
    """Report statistics on duplicate questions and answers."""
    q_counter = Counter(_normalize_text(s["question"]) for s in samples)
    a_counter = Counter(_normalize_text(s["answer"]) for s in samples)

    total = len(samples)

    # Questions that appear more than once
    dup_q = {q: c for q, c in q_counter.items() if c > 1}
    dup_q_count = sum(c for c in dup_q.values())  # total samples with dup questions
    unique_q = len(q_counter)

    # Answers that appear more than once
    dup_a = {a: c for a, c in a_counter.items() if c > 1}
    dup_a_count = sum(c for c in dup_a.values())  # total samples with dup answers
    unique_a = len(a_counter)

    print(f"\n{'='*70}")
    print(f"DUPLICATE ANALYSIS [{label}] ({total:,} samples)")
    print(f"{'='*70}")

    print(f"\n  Questions:")
    print(f"    Total:   {total:,}")
    print(f"    Unique:  {unique_q:,} ({unique_q/max(total,1)*100:.1f}%)")
    print(f"    Duplicated question strings: {len(dup_q):,}")
    print(f"    Samples with non-unique Q:   {dup_q_count:,} ({dup_q_count/max(total,1)*100:.1f}%)")

    if dup_q:
        print(f"    Top-10 most repeated questions:")
        for q_text, cnt in sorted(dup_q.items(), key=lambda x: -x[1])[:10]:
            display = q_text[:80] + "..." if len(q_text) > 80 else q_text
            print(f"      [{cnt:5,}x] {display}")

    print(f"\n  Answers:")
    print(f"    Total:   {total:,}")
    print(f"    Unique:  {unique_a:,} ({unique_a/max(total,1)*100:.1f}%)")
    print(f"    Duplicated answer strings: {len(dup_a):,}")
    print(f"    Samples with non-unique A:  {dup_a_count:,} ({dup_a_count/max(total,1)*100:.1f}%)")

    if dup_a:
        print(f"    Top-10 most repeated answers:")
        for a_text, cnt in sorted(dup_a.items(), key=lambda x: -x[1])[:10]:
            display = a_text[:80] + "..." if len(a_text) > 80 else a_text
            print(f"      [{cnt:5,}x] {display}")

    print()


# ========== Main download function ==========

def prepare_large_qa_data(test_mode: bool = False, only_chinese: bool = False,
                          only_english: bool = False, no_cache: bool = False):
    """Download, clean, and merge large-scale QA datasets.

    Supports resume: each dataset is cached after download. Re-running
    skips already-cached datasets. Use --no-cache to force re-download.
    """
    from datasets import load_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_all = []
    dev_all = []

    # Test mode limits
    max_train = 5000 if test_mode else 0
    max_dev = 500 if test_mode else 200  # keep dev small: ~200 per dataset

    total_datasets = 16
    idx = 0
    global_t0 = time.time()

    # Cache info
    if no_cache:
        cache_status = "DISABLED (--no-cache)"
    elif CACHE_DIR.exists():
        cached_files = list(CACHE_DIR.glob("*.json"))
        cache_status = f"ENABLED ({len(cached_files)} datasets cached, will resume)"
    else:
        cache_status = "ENABLED (empty, first run)"

    print("=" * 70, flush=True)
    print(f"Preparing Large-Scale QA Data ({'TEST MODE' if test_mode else 'FULL MODE'})", flush=True)
    print(f"Target: {total_datasets} datasets", flush=True)
    print(f"Cache: {cache_status}", flush=True)
    print(f"Cache dir: {CACHE_DIR}", flush=True)
    print("=" * 70, flush=True)
    print(flush=True)

    skip_chinese = only_english
    skip_english = only_chinese

    # ========== English Datasets ==========

    if not skip_english:
        # 1. SQuAD v1.1
        idx += 1
        print(f"[{idx}/{total_datasets}] SQuAD v1.1", flush=True)
        if not no_cache and _try_load_cache("squad_v1", train_all, dev_all):
            pass
        else:
            print("  Loading ...", flush=True)
            try:
                with _Timer("SQuAD v1"):
                    print("    downloading train split ...", flush=True)
                    sq1_train = _convert_squad(load_dataset("rajpurkar/squad", split="train"),
                                               max_train, source="squad_v1")
                    print("    downloading validation split ...", flush=True)
                    sq1_dev = _convert_squad(load_dataset("rajpurkar/squad", split="validation"),
                                             max_dev, source="squad_v1")
                    train_all.extend(sq1_train)
                    dev_all.extend(sq1_dev)
                    print(f"  OK SQuAD v1: train={len(sq1_train):,}, dev={len(sq1_dev):,}", flush=True)
                    _cache_save("squad_v1", sq1_train, sq1_dev)
            except Exception as e:
                print(f"  FAIL SQuAD v1: {e}", flush=True)
                traceback.print_exc()

        # 2. SQuAD v2.0 (with unanswerable - will be removed in cleaning)
        idx += 1
        print(f"[{idx}/{total_datasets}] SQuAD v2.0", flush=True)
        if not no_cache and _try_load_cache("squad_v2", train_all, dev_all):
            pass
        else:
            print("  Loading ...", flush=True)
            try:
                with _Timer("SQuAD v2"):
                    print("    downloading train split ...", flush=True)
                    sq2_train = _convert_squad(load_dataset("rajpurkar/squad_v2", split="train"),
                                               max_train, source="squad_v2",
                                               include_unanswerable=True)
                    print("    downloading validation split ...", flush=True)
                    sq2_dev = _convert_squad(load_dataset("rajpurkar/squad_v2", split="validation"),
                                             max_dev, source="squad_v2",
                                             include_unanswerable=True)
                    train_all.extend(sq2_train)
                    dev_all.extend(sq2_dev)
                    print(f"  OK SQuAD v2: train={len(sq2_train):,}, dev={len(sq2_dev):,}", flush=True)
                    _cache_save("squad_v2", sq2_train, sq2_dev)
            except Exception as e:
                print(f"  FAIL SQuAD v2: {e}", flush=True)
                traceback.print_exc()

        # 3. TriviaQA
        idx += 1
        print(f"[{idx}/{total_datasets}] TriviaQA", flush=True)
        if not no_cache and _try_load_cache("triviaqa", train_all, dev_all):
            pass
        else:
            print("  Loading (streaming) ...", flush=True)
            try:
                with _Timer("TriviaQA"):
                    print("    streaming train split ...", flush=True)
                    tqa_train = _convert_triviaqa(
                        _iter_with_progress(
                            load_dataset("trivia_qa", "rc", split="train", streaming=True),
                            desc="TriviaQA train", every=10000),
                        max_train, source="triviaqa")
                    print("    streaming validation split ...", flush=True)
                    tqa_dev = _convert_triviaqa(
                        _iter_with_progress(
                            load_dataset("trivia_qa", "rc", split="validation", streaming=True),
                            desc="TriviaQA dev", every=5000),
                        max_dev, source="triviaqa")
                    train_all.extend(tqa_train)
                    dev_all.extend(tqa_dev)
                    print(f"  OK TriviaQA: train={len(tqa_train):,}, dev={len(tqa_dev):,}", flush=True)
                    _cache_save("triviaqa", tqa_train, tqa_dev)
            except Exception as e:
                print(f"  FAIL TriviaQA: {e}", flush=True)
                traceback.print_exc()

        # 4. Natural Questions (no cap — conversion filters for short answers)
        idx += 1
        if not test_mode:
            print(f"[{idx}/{total_datasets}] Natural Questions", flush=True)
            if not no_cache and _try_load_cache("natural_questions", train_all, dev_all):
                pass
            else:
                print("  Loading (streaming, ~307K rows, may take 10-30 min) ...", flush=True)
                try:
                    with _Timer("Natural Questions"):
                        print("    streaming train split (287 parquet files) ...", flush=True)
                        nq_train = _convert_natural_questions(
                            _iter_with_progress(
                                load_dataset("natural_questions", split="train", streaming=True),
                                desc="NQ train", every=5000),
                            max_n=0, source="natural_questions")
                        print(f"    train done: {len(nq_train):,} samples kept", flush=True)
                        print("    streaming validation split ...", flush=True)
                        nq_dev = _convert_natural_questions(
                            _iter_with_progress(
                                load_dataset("natural_questions", split="validation", streaming=True),
                                desc="NQ dev", every=1000),
                            max_dev, source="natural_questions")
                        train_all.extend(nq_train)
                        dev_all.extend(nq_dev)
                        print(f"  OK Natural Questions: train={len(nq_train):,}, dev={len(nq_dev):,}", flush=True)
                        _cache_save("natural_questions", nq_train, nq_dev)
                except Exception as e:
                    print(f"  FAIL Natural Questions: {e}", flush=True)
                    traceback.print_exc()
        else:
            print(f"[{idx}/{total_datasets}] Natural Questions: skipped in test mode", flush=True)

        # 5. HotpotQA
        idx += 1
        print(f"[{idx}/{total_datasets}] HotpotQA", flush=True)
        if not no_cache and _try_load_cache("hotpotqa", train_all, dev_all):
            pass
        else:
            print("  Loading ...", flush=True)
            try:
                with _Timer("HotpotQA"):
                    print("    downloading train split ...", flush=True)
                    hpqa_train = _convert_hotpotqa(
                        load_dataset("hotpot_qa", "distractor", split="train"),
                        max_train, source="hotpotqa")
                    print("    downloading validation split ...", flush=True)
                    hpqa_dev = _convert_hotpotqa(
                        load_dataset("hotpot_qa", "distractor", split="validation"),
                        max_dev, source="hotpotqa")
                    train_all.extend(hpqa_train)
                    dev_all.extend(hpqa_dev)
                    print(f"  OK HotpotQA: train={len(hpqa_train):,}, dev={len(hpqa_dev):,}", flush=True)
                    _cache_save("hotpotqa", hpqa_train, hpqa_dev)
            except Exception as e:
                print(f"  FAIL HotpotQA: {e}", flush=True)
                traceback.print_exc()

        # 6. QuAC (conversational QA)
        # NOTE: allenai/quac uses legacy dataset scripts, no parquet alternative.
        # Blocked by datasets>=4.x. Skip gracefully.
        idx += 1
        print(f"[{idx}/{total_datasets}] QuAC: SKIPPED (no parquet-format repo available, "
              "allenai/quac uses legacy dataset scripts blocked by datasets>=4.x)")

        # 7. DROP (discrete reasoning)
        idx += 1
        print(f"[{idx}/{total_datasets}] DROP", flush=True)
        if not no_cache and _try_load_cache("drop", train_all, dev_all):
            pass
        else:
            print("  Loading ...", flush=True)
            try:
                with _Timer("DROP"):
                    drop_loaded = False
                    for repo in ["ucinlp/drop", "drop"]:
                        try:
                            print(f"    trying repo: {repo} ...", flush=True)
                            drop_train = _convert_drop(
                                load_dataset(repo, split="train"),
                                max_train, source="drop")
                            drop_dev = _convert_drop(
                                load_dataset(repo, split="validation"),
                                max_dev, source="drop")
                            train_all.extend(drop_train)
                            dev_all.extend(drop_dev)
                            print(f"  OK DROP: train={len(drop_train):,}, dev={len(drop_dev):,}", flush=True)
                            _cache_save("drop", drop_train, drop_dev)
                            drop_loaded = True
                            break
                        except Exception:
                            continue
                    if not drop_loaded:
                        print(f"  FAIL DROP: all repos failed", flush=True)
            except Exception as e:
                print(f"  FAIL DROP: {e}", flush=True)
                traceback.print_exc()

        # 8. AdversarialQA (SQuAD-format adversarial examples)
        idx += 1
        print(f"[{idx}/{total_datasets}] AdversarialQA", flush=True)
        if not no_cache and _try_load_cache("adversarial_qa", train_all, dev_all):
            pass
        else:
            print("  Loading ...", flush=True)
            try:
                with _Timer("AdversarialQA"):
                    adv_loaded = False
                    for config_name in ["adversarialQA", "dbpedia_based_adversarialQA",
                                        "squad_adversarialQA"]:
                        try:
                            print(f"    trying config: {config_name} ...", flush=True)
                            adv_train = _convert_squad(
                                load_dataset("adversarial_qa", config_name, split="train"),
                                max_train, source="adversarial_qa")
                            adv_dev = _convert_squad(
                                load_dataset("adversarial_qa", config_name, split="validation"),
                                max_dev, source="adversarial_qa")
                            train_all.extend(adv_train)
                            dev_all.extend(adv_dev)
                            print(f"  OK AdversarialQA ({config_name}): "
                                  f"train={len(adv_train):,}, dev={len(adv_dev):,}", flush=True)
                            _cache_save("adversarial_qa", adv_train, adv_dev)
                            adv_loaded = True
                        except Exception as e2:
                            print(f"    - config '{config_name}' failed: {e2}", flush=True)
                            continue
                    if not adv_loaded:
                        # Try without config
                        try:
                            print(f"    trying without config ...", flush=True)
                            adv_train = _convert_squad(
                                load_dataset("adversarial_qa", split="train"),
                                max_train, source="adversarial_qa")
                            adv_dev = _convert_squad(
                                load_dataset("adversarial_qa", split="validation"),
                                max_dev, source="adversarial_qa")
                            train_all.extend(adv_train)
                            dev_all.extend(adv_dev)
                            print(f"  OK AdversarialQA: train={len(adv_train):,}, dev={len(adv_dev):,}", flush=True)
                            _cache_save("adversarial_qa", adv_train, adv_dev)
                        except Exception:
                            print(f"  FAIL AdversarialQA: all configs failed", flush=True)
            except Exception as e:
                print(f"  FAIL AdversarialQA: {e}", flush=True)
                traceback.print_exc()

        # 9. DuoRC (SelfRC — movie plot QA)
        idx += 1
        print(f"[{idx}/{total_datasets}] DuoRC", flush=True)
        if not no_cache and _try_load_cache("duorc", train_all, dev_all):
            pass
        else:
            print("  Loading ...", flush=True)
            try:
                with _Timer("DuoRC"):
                    duo_all_train, duo_all_dev = [], []
                    duo_loaded = False
                    for config_name in ["SelfRC", "ParaphraseRC"]:
                        try:
                            print(f"    trying config: {config_name} ...", flush=True)
                            duo_train = _convert_duorc(
                                load_dataset("duorc", config_name, split="train"),
                                max_train, source=f"duorc_{config_name.lower()}")
                            duo_dev = _convert_duorc(
                                load_dataset("duorc", config_name, split="validation"),
                                max_dev, source=f"duorc_{config_name.lower()}")
                            duo_all_train.extend(duo_train)
                            duo_all_dev.extend(duo_dev)
                            train_all.extend(duo_train)
                            dev_all.extend(duo_dev)
                            print(f"  OK DuoRC ({config_name}): "
                                  f"train={len(duo_train):,}, dev={len(duo_dev):,}", flush=True)
                            duo_loaded = True
                        except Exception as e2:
                            print(f"    - config '{config_name}' failed: {e2}", flush=True)
                            continue
                    if duo_loaded:
                        _cache_save("duorc", duo_all_train, duo_all_dev)
                    else:
                        print(f"  FAIL DuoRC: all configs failed", flush=True)
            except Exception as e:
                print(f"  FAIL DuoRC: {e}", flush=True)
                traceback.print_exc()
    else:
        idx += 9
        print(f"[1-{idx}/{total_datasets}] English datasets: skipped (only_chinese=True)")

    # ========== Chinese Datasets ==========

    if not skip_chinese:
        # 10. CMRC2018
        idx += 1
        print(f"[{idx}/{total_datasets}] CMRC2018", flush=True)
        if not no_cache and _try_load_cache("cmrc", train_all, dev_all):
            pass
        else:
            print("  Loading ...", flush=True)
            try:
                with _Timer("CMRC2018"):
                    print("    downloading train split ...", flush=True)
                    cm_train = _convert_cmrc_or_drcd(load_dataset("cmrc2018", split="train"),
                                                     max_train, source="cmrc")
                    print("    downloading validation split ...", flush=True)
                    cm_dev = _convert_cmrc_or_drcd(load_dataset("cmrc2018", split="validation"),
                                                   max_dev, source="cmrc")
                    train_all.extend(cm_train)
                    dev_all.extend(cm_dev)
                    print(f"  OK CMRC2018: train={len(cm_train):,}, dev={len(cm_dev):,}", flush=True)
                    _cache_save("cmrc", cm_train, cm_dev)
            except Exception as e:
                print(f"  FAIL CMRC2018: {e}", flush=True)
                traceback.print_exc()

        # 11. DuReader
        idx += 1
        print(f"[{idx}/{total_datasets}] DuReader", flush=True)
        if not no_cache and _try_load_cache("dureader", train_all, dev_all):
            pass
        else:
            print("  Loading ...", flush=True)
            try:
                with _Timer("DuReader"):
                    loaded = False
                    for repo in ["dirtycomputer/dureader_robust-data",
                                  "PaddlePaddle/dureader_robust", "luozhouyang/dureader"]:
                        try:
                            print(f"    trying repo: {repo} ...", flush=True)
                            d_train = _convert_dureader(load_dataset(repo, split="train"),
                                                        max_train, source="dureader")
                            d_dev = _convert_dureader(load_dataset(repo, split="validation"),
                                                      max_dev, source="dureader")
                            train_all.extend(d_train)
                            dev_all.extend(d_dev)
                            print(f"  OK DuReader: train={len(d_train):,}, dev={len(d_dev):,}", flush=True)
                            _cache_save("dureader", d_train, d_dev)
                            loaded = True
                            break
                        except Exception:
                            continue
                    if not loaded:
                        print(f"  FAIL DuReader: all repos failed", flush=True)
            except Exception as e:
                print(f"  FAIL DuReader: {e}", flush=True)
                traceback.print_exc()

        # 12. DRCD (Traditional Chinese)
        idx += 1
        print(f"[{idx}/{total_datasets}] DRCD", flush=True)
        if not no_cache and _try_load_cache("drcd", train_all, dev_all):
            pass
        else:
            print("  Loading ...", flush=True)
            try:
                with _Timer("DRCD"):
                    drcd_loaded = False
                    for repo in ["clue/drcd", "hfl/drcd"]:
                        try:
                            print(f"    trying repo: {repo} ...", flush=True)
                            if repo == "clue/drcd":
                                dr_train = _convert_cmrc_or_drcd(
                                    load_dataset("clue", "drcd", split="train"),
                                    max_train, source="drcd")
                                dr_dev = _convert_cmrc_or_drcd(
                                    load_dataset("clue", "drcd", split="validation"),
                                    max_dev, source="drcd")
                            else:
                                dr_train = _convert_cmrc_or_drcd(
                                    load_dataset("hfl/drcd", split="train"),
                                    max_train, source="drcd")
                                dr_dev = _convert_cmrc_or_drcd(
                                    load_dataset("hfl/drcd", split="validation"),
                                    max_dev, source="drcd")
                            train_all.extend(dr_train)
                            dev_all.extend(dr_dev)
                            print(f"  OK DRCD: train={len(dr_train):,}, dev={len(dr_dev):,}", flush=True)
                            _cache_save("drcd", dr_train, dr_dev)
                            drcd_loaded = True
                            break
                        except Exception:
                            continue
                    if not drcd_loaded:
                        print(f"  FAIL DRCD: all repos failed", flush=True)
            except Exception as e:
                print(f"  FAIL DRCD: {e}", flush=True)
                traceback.print_exc()

        # 13. WebQA (Chinese web)
        idx += 1
        print(f"[{idx}/{total_datasets}] WebQA", flush=True)
        if not no_cache and _try_load_cache("webqa", train_all, dev_all):
            pass
        else:
            print("  Loading ...", flush=True)
            try:
                with _Timer("WebQA"):
                    webqa_loaded = False
                    for repo in ["suolyer/webqa", "THUDM/webqa"]:
                        try:
                            print(f"    trying repo: {repo} ...", flush=True)
                            wq_train = _convert_webqa(load_dataset(repo, split="train"),
                                                      max_train, source="webqa")
                            try:
                                wq_dev = _convert_webqa(load_dataset(repo, split="validation"),
                                                        max_dev, source="webqa")
                            except Exception:
                                wq_dev = []
                            train_all.extend(wq_train)
                            dev_all.extend(wq_dev)
                            print(f"  OK WebQA: train={len(wq_train):,}, dev={len(wq_dev):,}", flush=True)
                            _cache_save("webqa", wq_train, wq_dev)
                            webqa_loaded = True
                            break
                        except Exception:
                            continue
                    if not webqa_loaded:
                        print(f"  FAIL WebQA: all repos failed or not available", flush=True)
            except Exception as e:
                print(f"  FAIL WebQA: {e}", flush=True)
                traceback.print_exc()
    else:
        idx += 4
        print(f"[10-{idx}/{total_datasets}] Chinese datasets: skipped (only_english=True)")

    # ========== Multilingual Datasets ==========

    # 14. MLQA (English + Chinese)
    # NOTE: facebook/mlqa uses legacy dataset scripts, blocked by datasets>=4.x.
    # Skip gracefully.
    idx += 1
    print(f"[{idx}/{total_datasets}] MLQA: SKIPPED (no parquet-format repo available, "
          "facebook/mlqa uses legacy dataset scripts blocked by datasets>=4.x)")

    # 15. XQuAD (English + Chinese)
    idx += 1
    print(f"[{idx}/{total_datasets}] XQuAD", flush=True)
    if not no_cache and _try_load_cache("xquad", train_all, dev_all):
        pass
    else:
        print("  Loading ...", flush=True)
        try:
            with _Timer("XQuAD"):
                xq_all_train, xq_all_dev = [], []
                xquad_loaded = False
                for repo in ["google/xquad", "xquad"]:
                    try:
                        print(f"    trying repo: {repo} ...", flush=True)
                        if not skip_english:
                            xq_en = _convert_squad(
                                load_dataset(repo, "xquad.en", split="validation"),
                                max_dev, source="xquad_en")
                            dev_all.extend(xq_en)
                            xq_all_dev.extend(xq_en)
                            print(f"  OK XQuAD-en: {len(xq_en):,}", flush=True)

                        if not skip_chinese:
                            xq_zh = _convert_squad(
                                load_dataset(repo, "xquad.zh", split="validation"),
                                max_dev, source="xquad_zh")
                            dev_all.extend(xq_zh)
                            xq_all_dev.extend(xq_zh)
                            print(f"  OK XQuAD-zh: {len(xq_zh):,}", flush=True)

                        xquad_loaded = True
                        break
                    except Exception as e2:
                        print(f"    - repo '{repo}' failed: {e2}", flush=True)
                        continue
                if xquad_loaded:
                    _cache_save("xquad", xq_all_train, xq_all_dev)
                else:
                    print(f"  FAIL XQuAD: all repos failed", flush=True)
        except Exception as e:
            print(f"  FAIL XQuAD: {e}", flush=True)
            traceback.print_exc()

    # 16. CoQA (conversational QA)
    idx += 1
    if not skip_english:
        print(f"[{idx}/{total_datasets}] CoQA", flush=True)
        if not no_cache and _try_load_cache("coqa", train_all, dev_all):
            pass
        else:
            print("  Loading ...", flush=True)
            try:
                with _Timer("CoQA"):
                    coqa_loaded = False
                    coqa_all_train, coqa_all_dev = [], []
                    for repo in ["stanfordnlp/coqa", "coqa"]:
                        try:
                            print(f"    trying repo: {repo} ...", flush=True)
                            coqa_train = _convert_coqa(
                                load_dataset(repo, split="train"),
                                max_train, source="coqa")
                            coqa_all_train.extend(coqa_train)
                            train_all.extend(coqa_train)
                            print(f"  OK CoQA: train={len(coqa_train):,}", flush=True)
                            try:
                                coqa_dev = _convert_coqa(
                                    load_dataset(repo, split="validation"),
                                    max_dev, source="coqa")
                                coqa_all_dev.extend(coqa_dev)
                                dev_all.extend(coqa_dev)
                                print(f"  OK CoQA dev: {len(coqa_dev):,}", flush=True)
                            except Exception:
                                pass
                            coqa_loaded = True
                            break
                        except Exception as e2:
                            print(f"    - repo '{repo}' failed: {e2}", flush=True)
                            continue
                    if coqa_loaded:
                        _cache_save("coqa", coqa_all_train, coqa_all_dev)
                    else:
                        print(f"  FAIL CoQA: all repos failed (may require manual download)", flush=True)
            except Exception as e:
                print(f"  FAIL CoQA: {e}", flush=True)
                traceback.print_exc()
    else:
        print(f"[{idx}/{total_datasets}] CoQA: skipped (only_chinese=True)", flush=True)

    # ========== Data Cleaning ==========

    print()
    print("=" * 70)
    print("DATA CLEANING")
    print("=" * 70)

    pre_clean_train = len(train_all)
    pre_clean_dev = len(dev_all)

    train_all = clean_data(train_all, label="train")
    dev_all = clean_data(dev_all, label="dev")

    print(f"\n  Total cleaned: train {pre_clean_train:,} -> {len(train_all):,}, "
          f"dev {pre_clean_dev:,} -> {len(dev_all):,}")

    # ========== Duplicate Analysis ==========

    report_duplicates(train_all, label="train")
    report_duplicates(dev_all, label="dev")

    # ========== Shuffle and Save ==========

    print("=" * 70)
    print("SAVING")
    print("=" * 70)

    random.seed(42)
    random.shuffle(train_all)
    random.shuffle(dev_all)

    train_path = DATA_DIR / "qa_large_train.json"
    dev_path = DATA_DIR / "qa_large_dev.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_all, f, ensure_ascii=False, indent=None)

    with open(dev_path, "w", encoding="utf-8") as f:
        json.dump(dev_all, f, ensure_ascii=False, indent=None)

    # Summary
    print(f"\n  Train: {len(train_all):,} samples -> {train_path}")
    print(f"  Size: {train_path.stat().st_size / 1e6:.1f} MB")
    print(f"\n  Dev:   {len(dev_all):,} samples -> {dev_path}")
    print(f"  Size: {dev_path.stat().st_size / 1e6:.1f} MB")

    # Source breakdown
    print(f"\n{'='*70}")
    print("SOURCE BREAKDOWN (train)")
    print(f"{'='*70}")
    train_sources = Counter(s.get("source", "unknown") for s in train_all)
    for src, count in train_sources.most_common():
        pct = 100 * count / len(train_all)
        print(f"  {src:25s}: {count:8,} ({pct:5.1f}%)")

    print(f"\n{'='*70}")
    print("SOURCE BREAKDOWN (dev)")
    print(f"{'='*70}")
    dev_sources = Counter(s.get("source", "unknown") for s in dev_all)
    for src, count in dev_sources.most_common():
        pct = 100 * count / len(dev_all)
        print(f"  {src:25s}: {count:8,} ({pct:5.1f}%)")

    total_elapsed = time.time() - global_t0
    minutes = int(total_elapsed // 60)
    seconds = int(total_elapsed % 60)
    print(f"\n{'='*70}", flush=True)
    print(f"DONE - Large-scale QA data preparation complete! (total: {minutes}m {seconds}s)", flush=True)
    print(f"{'='*70}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare large-scale QA datasets for pure QA training"
    )
    parser.add_argument("--test", action="store_true",
                        help="Test mode: download small subsets (~5K train, ~1K dev per dataset)")
    parser.add_argument("--only-chinese", action="store_true",
                        help="Download only Chinese datasets")
    parser.add_argument("--only-english", action="store_true",
                        help="Download only English datasets")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force re-download all datasets (ignore cache)")
    parser.add_argument("--clear-cache", action="store_true",
                        help="Delete all cached datasets and exit")

    args = parser.parse_args()

    if args.clear_cache:
        clear_cache()
        return

    if args.only_chinese and args.only_english:
        print("ERROR: Cannot specify both --only-chinese and --only-english")
        return

    prepare_large_qa_data(
        test_mode=args.test,
        only_chinese=args.only_chinese,
        only_english=args.only_english,
        no_cache=args.no_cache,
    )


if __name__ == "__main__":
    main()
