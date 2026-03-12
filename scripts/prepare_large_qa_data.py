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
    """Convert DuReader-robust dataset."""
    samples = []
    for row in dataset:
        if max_n and len(samples) >= max_n:
            break
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
    """
    samples = []
    for row in dataset:
        if max_n and len(samples) >= max_n:
            break

        short_answers = row.get("annotations", {}).get("short_answers", [])
        if not short_answers or not short_answers[0]:
            continue

        answer_start = short_answers[0].get("start_token", -1)
        answer_end = short_answers[0].get("end_token", -1)
        if answer_start < 0 or answer_end < 0:
            continue

        context = row.get("document", {}).get("text", "")
        if not context:
            tokens = row.get("document", {}).get("tokens", [])
            if tokens:
                context = " ".join(t.get("token", "") for t in tokens)

        if not context or len(context) < 100:
            continue

        if len(context) > max_context_chars:
            context = context[:max_context_chars]

        tokens = row.get("document", {}).get("tokens", [])
        if tokens:
            answer = " ".join(t.get("token", "") for t in tokens[answer_start:answer_end])
        else:
            answer = context[answer_start:answer_end]

        if not answer.strip():
            continue

        samples.append({
            "context": context,
            "question": row["question"]["text"],
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
                          only_english: bool = False):
    """Download, clean, and merge large-scale QA datasets."""
    from datasets import load_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_all = []
    dev_all = []

    # Test mode limits
    max_train = 5000 if test_mode else 0
    max_dev = 500 if test_mode else 200  # keep dev small: ~200 per dataset

    total_datasets = 16
    idx = 0

    print("=" * 70)
    print(f"Preparing Large-Scale QA Data ({'TEST MODE' if test_mode else 'FULL MODE'})")
    print(f"Target: {total_datasets} datasets")
    print("=" * 70)
    print()

    skip_chinese = only_english
    skip_english = only_chinese

    # ========== English Datasets ==========

    if not skip_english:
        # 1. SQuAD v1.1
        idx += 1
        print(f"[{idx}/{total_datasets}] Loading SQuAD v1.1 ...")
        try:
            sq1_train = _convert_squad(load_dataset("rajpurkar/squad", split="train"),
                                       max_train, source="squad_v1")
            sq1_dev = _convert_squad(load_dataset("rajpurkar/squad", split="validation"),
                                     max_dev, source="squad_v1")
            train_all.extend(sq1_train)
            dev_all.extend(sq1_dev)
            print(f"  OK SQuAD v1: train={len(sq1_train):,}, dev={len(sq1_dev):,}")
        except Exception as e:
            print(f"  FAIL SQuAD v1: {e}")
            traceback.print_exc()

        # 2. SQuAD v2.0 (with unanswerable - will be removed in cleaning)
        idx += 1
        print(f"[{idx}/{total_datasets}] Loading SQuAD v2.0 ...")
        try:
            sq2_train = _convert_squad(load_dataset("rajpurkar/squad_v2", split="train"),
                                       max_train, source="squad_v2",
                                       include_unanswerable=True)
            sq2_dev = _convert_squad(load_dataset("rajpurkar/squad_v2", split="validation"),
                                     max_dev, source="squad_v2",
                                     include_unanswerable=True)
            train_all.extend(sq2_train)
            dev_all.extend(sq2_dev)
            print(f"  OK SQuAD v2: train={len(sq2_train):,}, dev={len(sq2_dev):,}")
        except Exception as e:
            print(f"  FAIL SQuAD v2: {e}")
            traceback.print_exc()

        # 3. TriviaQA
        idx += 1
        print(f"[{idx}/{total_datasets}] Loading TriviaQA (streaming) ...")
        try:
            tqa_train = _convert_triviaqa(
                load_dataset("trivia_qa", "rc", split="train", streaming=True),
                max_train, source="triviaqa")
            tqa_dev = _convert_triviaqa(
                load_dataset("trivia_qa", "rc", split="validation", streaming=True),
                max_dev, source="triviaqa")
            train_all.extend(tqa_train)
            dev_all.extend(tqa_dev)
            print(f"  OK TriviaQA: train={len(tqa_train):,}, dev={len(tqa_dev):,}")
        except Exception as e:
            print(f"  FAIL TriviaQA: {e}")
            traceback.print_exc()

        # 4. Natural Questions (no cap — conversion filters for short answers)
        idx += 1
        if not test_mode:
            print(f"[{idx}/{total_datasets}] Loading Natural Questions (streaming, may take time) ...")
            try:
                nq_train = _convert_natural_questions(
                    load_dataset("natural_questions", split="train", streaming=True),
                    max_n=0, source="natural_questions")
                nq_dev = _convert_natural_questions(
                    load_dataset("natural_questions", split="validation", streaming=True),
                    max_dev, source="natural_questions")
                train_all.extend(nq_train)
                dev_all.extend(nq_dev)
                print(f"  OK Natural Questions: train={len(nq_train):,}, dev={len(nq_dev):,}")
            except Exception as e:
                print(f"  FAIL Natural Questions: {e}")
                traceback.print_exc()
        else:
            print(f"[{idx}/{total_datasets}] Natural Questions: skipped in test mode")

        # 5. HotpotQA
        idx += 1
        print(f"[{idx}/{total_datasets}] Loading HotpotQA ...")
        try:
            hpqa_train = _convert_hotpotqa(
                load_dataset("hotpot_qa", "distractor", split="train"),
                max_train, source="hotpotqa")
            hpqa_dev = _convert_hotpotqa(
                load_dataset("hotpot_qa", "distractor", split="validation"),
                max_dev, source="hotpotqa")
            train_all.extend(hpqa_train)
            dev_all.extend(hpqa_dev)
            print(f"  OK HotpotQA: train={len(hpqa_train):,}, dev={len(hpqa_dev):,}")
        except Exception as e:
            print(f"  FAIL HotpotQA: {e}")
            traceback.print_exc()

        # 6. QuAC (conversational QA)
        idx += 1
        print(f"[{idx}/{total_datasets}] Loading QuAC ...")
        try:
            quac_train = _convert_quac(
                load_dataset("quac", split="train"),
                max_train, source="quac")
            quac_dev = _convert_quac(
                load_dataset("quac", split="validation"),
                max_dev, source="quac")
            train_all.extend(quac_train)
            dev_all.extend(quac_dev)
            print(f"  OK QuAC: train={len(quac_train):,}, dev={len(quac_dev):,}")
        except Exception as e:
            print(f"  FAIL QuAC: {e}")
            traceback.print_exc()

        # 7. DROP (discrete reasoning)
        idx += 1
        print(f"[{idx}/{total_datasets}] Loading DROP ...")
        try:
            drop_loaded = False
            for repo in ["ucinlp/drop", "drop"]:
                try:
                    drop_train = _convert_drop(
                        load_dataset(repo, split="train"),
                        max_train, source="drop")
                    drop_dev = _convert_drop(
                        load_dataset(repo, split="validation"),
                        max_dev, source="drop")
                    train_all.extend(drop_train)
                    dev_all.extend(drop_dev)
                    print(f"  OK DROP: train={len(drop_train):,}, dev={len(drop_dev):,}")
                    drop_loaded = True
                    break
                except Exception:
                    continue
            if not drop_loaded:
                print(f"  FAIL DROP: all repos failed")
        except Exception as e:
            print(f"  FAIL DROP: {e}")
            traceback.print_exc()

        # 8. AdversarialQA (SQuAD-format adversarial examples)
        idx += 1
        print(f"[{idx}/{total_datasets}] Loading AdversarialQA ...")
        try:
            adv_loaded = False
            for config_name in ["adversarialQA", "dbpedia_based_adversarialQA",
                                "squad_adversarialQA"]:
                try:
                    adv_train = _convert_squad(
                        load_dataset("adversarial_qa", config_name, split="train"),
                        max_train, source="adversarial_qa")
                    adv_dev = _convert_squad(
                        load_dataset("adversarial_qa", config_name, split="validation"),
                        max_dev, source="adversarial_qa")
                    train_all.extend(adv_train)
                    dev_all.extend(adv_dev)
                    print(f"  OK AdversarialQA ({config_name}): "
                          f"train={len(adv_train):,}, dev={len(adv_dev):,}")
                    adv_loaded = True
                except Exception as e2:
                    print(f"    - config '{config_name}' failed: {e2}")
                    continue
            if not adv_loaded:
                # Try without config
                try:
                    adv_train = _convert_squad(
                        load_dataset("adversarial_qa", split="train"),
                        max_train, source="adversarial_qa")
                    adv_dev = _convert_squad(
                        load_dataset("adversarial_qa", split="validation"),
                        max_dev, source="adversarial_qa")
                    train_all.extend(adv_train)
                    dev_all.extend(adv_dev)
                    print(f"  OK AdversarialQA: train={len(adv_train):,}, dev={len(adv_dev):,}")
                except Exception:
                    print(f"  FAIL AdversarialQA: all configs failed")
        except Exception as e:
            print(f"  FAIL AdversarialQA: {e}")
            traceback.print_exc()

        # 9. DuoRC (SelfRC — movie plot QA)
        idx += 1
        print(f"[{idx}/{total_datasets}] Loading DuoRC (SelfRC) ...")
        try:
            duo_loaded = False
            for config_name in ["SelfRC", "ParaphraseRC"]:
                try:
                    duo_train = _convert_duorc(
                        load_dataset("duorc", config_name, split="train"),
                        max_train, source=f"duorc_{config_name.lower()}")
                    duo_dev = _convert_duorc(
                        load_dataset("duorc", config_name, split="validation"),
                        max_dev, source=f"duorc_{config_name.lower()}")
                    train_all.extend(duo_train)
                    dev_all.extend(duo_dev)
                    print(f"  OK DuoRC ({config_name}): "
                          f"train={len(duo_train):,}, dev={len(duo_dev):,}")
                    duo_loaded = True
                except Exception as e2:
                    print(f"    - config '{config_name}' failed: {e2}")
                    continue
            if not duo_loaded:
                print(f"  FAIL DuoRC: all configs failed")
        except Exception as e:
            print(f"  FAIL DuoRC: {e}")
            traceback.print_exc()
    else:
        idx += 9
        print(f"[1-{idx}/{total_datasets}] English datasets: skipped (only_chinese=True)")

    # ========== Chinese Datasets ==========

    if not skip_chinese:
        # 10. CMRC2018
        idx += 1
        print(f"[{idx}/{total_datasets}] Loading CMRC2018 ...")
        try:
            cm_train = _convert_cmrc_or_drcd(load_dataset("cmrc2018", split="train"),
                                             max_train, source="cmrc")
            cm_dev = _convert_cmrc_or_drcd(load_dataset("cmrc2018", split="validation"),
                                           max_dev, source="cmrc")
            train_all.extend(cm_train)
            dev_all.extend(cm_dev)
            print(f"  OK CMRC2018: train={len(cm_train):,}, dev={len(cm_dev):,}")
        except Exception as e:
            print(f"  FAIL CMRC2018: {e}")
            traceback.print_exc()

        # 11. DuReader
        idx += 1
        print(f"[{idx}/{total_datasets}] Loading DuReader ...")
        try:
            loaded = False
            for repo in ["PaddlePaddle/dureader_robust", "luozhouyang/dureader"]:
                try:
                    d_train = _convert_dureader(load_dataset(repo, split="train"),
                                                max_train, source="dureader")
                    d_dev = _convert_dureader(load_dataset(repo, split="validation"),
                                              max_dev, source="dureader")
                    train_all.extend(d_train)
                    dev_all.extend(d_dev)
                    print(f"  OK DuReader: train={len(d_train):,}, dev={len(d_dev):,}")
                    loaded = True
                    break
                except Exception:
                    continue
            if not loaded:
                print(f"  FAIL DuReader: all repos failed")
        except Exception as e:
            print(f"  FAIL DuReader: {e}")
            traceback.print_exc()

        # 12. DRCD (Traditional Chinese)
        idx += 1
        print(f"[{idx}/{total_datasets}] Loading DRCD ...")
        try:
            drcd_loaded = False
            for repo in ["clue/drcd", "hfl/drcd"]:
                try:
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
                    print(f"  OK DRCD: train={len(dr_train):,}, dev={len(dr_dev):,}")
                    drcd_loaded = True
                    break
                except Exception:
                    continue
            if not drcd_loaded:
                print(f"  FAIL DRCD: all repos failed")
        except Exception as e:
            print(f"  FAIL DRCD: {e}")
            traceback.print_exc()

        # 13. WebQA (Chinese web)
        idx += 1
        print(f"[{idx}/{total_datasets}] Loading WebQA ...")
        try:
            webqa_loaded = False
            for repo in ["suolyer/webqa", "THUDM/webqa"]:
                try:
                    wq_train = _convert_webqa(load_dataset(repo, split="train"),
                                              max_train, source="webqa")
                    try:
                        wq_dev = _convert_webqa(load_dataset(repo, split="validation"),
                                                max_dev, source="webqa")
                    except Exception:
                        wq_dev = []
                    train_all.extend(wq_train)
                    dev_all.extend(wq_dev)
                    print(f"  OK WebQA: train={len(wq_train):,}, dev={len(wq_dev):,}")
                    webqa_loaded = True
                    break
                except Exception:
                    continue
            if not webqa_loaded:
                print(f"  FAIL WebQA: all repos failed or not available")
        except Exception as e:
            print(f"  FAIL WebQA: {e}")
            traceback.print_exc()
    else:
        idx += 4
        print(f"[10-{idx}/{total_datasets}] Chinese datasets: skipped (only_english=True)")

    # ========== Multilingual Datasets ==========

    # 14. MLQA (English + Chinese)
    idx += 1
    print(f"[{idx}/{total_datasets}] Loading MLQA ...")
    try:
        mlqa_loaded = False
        for repo in ["facebook/mlqa", "mlqa"]:
            try:
                if not skip_english:
                    # English subset
                    mlqa_en_test = _convert_squad(
                        load_dataset(repo, "mlqa.en.en", split="test"),
                        max_dev, source="mlqa_en")
                    dev_all.extend(mlqa_en_test)
                    print(f"  OK MLQA-en: test={len(mlqa_en_test):,}")
                    try:
                        mlqa_en_val = _convert_squad(
                            load_dataset(repo, "mlqa.en.en", split="validation"),
                            max_dev, source="mlqa_en")
                        dev_all.extend(mlqa_en_val)
                        print(f"  OK MLQA-en val: {len(mlqa_en_val):,}")
                    except Exception:
                        pass

                if not skip_chinese:
                    # Chinese subset
                    mlqa_zh_test = _convert_squad(
                        load_dataset(repo, "mlqa.zh.zh", split="test"),
                        max_dev, source="mlqa_zh")
                    dev_all.extend(mlqa_zh_test)
                    print(f"  OK MLQA-zh: test={len(mlqa_zh_test):,}")
                    try:
                        mlqa_zh_val = _convert_squad(
                            load_dataset(repo, "mlqa.zh.zh", split="validation"),
                            max_dev, source="mlqa_zh")
                        dev_all.extend(mlqa_zh_val)
                        print(f"  OK MLQA-zh val: {len(mlqa_zh_val):,}")
                    except Exception:
                        pass

                mlqa_loaded = True
                break
            except Exception as e2:
                print(f"    - repo '{repo}' failed: {e2}")
                continue
        if not mlqa_loaded:
            print(f"  FAIL MLQA: all repos failed")
    except Exception as e:
        print(f"  FAIL MLQA: {e}")
        traceback.print_exc()

    # 15. XQuAD (English + Chinese)
    idx += 1
    print(f"[{idx}/{total_datasets}] Loading XQuAD ...")
    try:
        xquad_loaded = False
        for repo in ["google/xquad", "xquad"]:
            try:
                if not skip_english:
                    xq_en = _convert_squad(
                        load_dataset(repo, "xquad.en", split="validation"),
                        max_dev, source="xquad_en")
                    dev_all.extend(xq_en)
                    print(f"  OK XQuAD-en: {len(xq_en):,}")

                if not skip_chinese:
                    xq_zh = _convert_squad(
                        load_dataset(repo, "xquad.zh", split="validation"),
                        max_dev, source="xquad_zh")
                    dev_all.extend(xq_zh)
                    print(f"  OK XQuAD-zh: {len(xq_zh):,}")

                xquad_loaded = True
                break
            except Exception as e2:
                print(f"    - repo '{repo}' failed: {e2}")
                continue
        if not xquad_loaded:
            print(f"  FAIL XQuAD: all repos failed")
    except Exception as e:
        print(f"  FAIL XQuAD: {e}")
        traceback.print_exc()

    # 16. CoQA (conversational QA)
    idx += 1
    if not skip_english:
        print(f"[{idx}/{total_datasets}] Loading CoQA ...")
        try:
            coqa_loaded = False
            for repo in ["stanfordnlp/coqa", "coqa"]:
                try:
                    coqa_train = _convert_coqa(
                        load_dataset(repo, split="train"),
                        max_train, source="coqa")
                    train_all.extend(coqa_train)
                    print(f"  OK CoQA: train={len(coqa_train):,}")
                    try:
                        coqa_dev = _convert_coqa(
                            load_dataset(repo, split="validation"),
                            max_dev, source="coqa")
                        dev_all.extend(coqa_dev)
                        print(f"  OK CoQA dev: {len(coqa_dev):,}")
                    except Exception:
                        pass
                    coqa_loaded = True
                    break
                except Exception as e2:
                    print(f"    - repo '{repo}' failed: {e2}")
                    continue
            if not coqa_loaded:
                print(f"  FAIL CoQA: all repos failed (may require manual download)")
        except Exception as e:
            print(f"  FAIL CoQA: {e}")
            traceback.print_exc()
    else:
        print(f"[{idx}/{total_datasets}] CoQA: skipped (only_chinese=True)")

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

    print(f"\n{'='*70}")
    print("DONE - Large-scale QA data preparation complete!")
    print(f"{'='*70}")


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

    args = parser.parse_args()

    if args.only_chinese and args.only_english:
        print("ERROR: Cannot specify both --only-chinese and --only-english")
        return

    prepare_large_qa_data(
        test_mode=args.test,
        only_chinese=args.only_chinese,
        only_english=args.only_english,
    )


if __name__ == "__main__":
    main()
