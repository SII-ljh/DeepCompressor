#!/usr/bin/env python3
"""Prepare large-scale QA datasets for pure QA training (without Stage 1 NTP).

This script downloads significantly more QA data than prepare_data.py to support
training models directly on QA tasks without NTP pretraining.

Target size:
  - Full mode:  ~500K-1M train samples, ~50K dev samples
  - Test mode:  ~10K train, ~2K dev

Datasets included:
  1. SQuAD v1.1 (87K train, 10K dev) - English
  2. SQuAD v2.0 (130K train, 12K dev) - English with unanswerable
  3. CMRC2018 (10K train, 3K dev) - Chinese
  4. DuReader (15K train, 1.4K dev) - Chinese
  5. DRCD (27K train, 4K dev) - Traditional Chinese
  6. TriviaQA (95K train, 12K dev) - English
  7. Natural Questions (307K train, 8K dev) - English, long context
  8. HotpotQA (90K train, 7.4K dev) - Multi-hop reasoning
  9. WebQA (42K train) - Chinese web QA

Total: ~800K train, ~60K dev (approximate)

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
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"


# ========== Conversion functions ==========

def _convert_squad(dataset, max_n: int = 0, source: str = "squad", include_unanswerable: bool = False):
    """Convert SQuAD-format dataset."""
    samples = []
    for row in dataset:
        if max_n and len(samples) >= max_n:
            break
        answers = row["answers"]["text"]

        # Handle unanswerable questions in SQuAD v2.0
        if not answers:
            if include_unanswerable:
                # For unanswerable, use a standard marker
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

        # Cap context length
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

    Only keeps samples with short answers (ignores long-answer-only and no-answer samples).
    """
    samples = []
    for row in dataset:
        if max_n and len(samples) >= max_n:
            break

        # Get short answers
        short_answers = row.get("annotations", {}).get("short_answers", [])
        if not short_answers or not short_answers[0]:
            continue

        # Extract first short answer text
        answer_start = short_answers[0].get("start_token", -1)
        answer_end = short_answers[0].get("end_token", -1)

        if answer_start < 0 or answer_end < 0:
            continue

        # Get context (document_text or tokens)
        context = row.get("document", {}).get("text", "")
        if not context:
            tokens = row.get("document", {}).get("tokens", [])
            if tokens:
                context = " ".join(t.get("token", "") for t in tokens)

        if not context or len(context) < 100:
            continue

        # Cap context
        if len(context) > max_context_chars:
            context = context[:max_context_chars]

        # Extract answer text from context
        tokens = row.get("document", {}).get("tokens", [])
        if tokens:
            answer = " ".join(t.get("token", "") for t in tokens[answer_start:answer_end])
        else:
            # Fallback: use character positions (approximate)
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

        # Concatenate supporting contexts
        contexts = row.get("context", {}).get("sentences", [])
        if not contexts:
            # Fallback to title-based contexts
            titles = row.get("context", {}).get("title", [])
            contexts = titles

        if not contexts:
            continue

        # Join contexts
        if isinstance(contexts[0], list):
            # Nested structure: [[sent1, sent2], [sent3, sent4]]
            context = " ".join(" ".join(sents) for sents in contexts)
        else:
            context = " ".join(contexts)

        if len(context) < 100:
            continue

        # Cap context
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

        # Get answer
        answers = row.get("answers", [])
        if not answers:
            answer = row.get("answer", "").strip()
            if not answer:
                continue
        else:
            answer = answers[0] if isinstance(answers, list) else answers

        # Get context
        context = row.get("evidence", "")
        if not context:
            context = row.get("context", "")

        if not context or len(context) < 50:
            continue

        # Cap context
        if len(context) > max_context_chars:
            context = context[:max_context_chars]

        samples.append({
            "context": context,
            "question": row.get("question", ""),
            "answer": answer,
            "source": source,
        })

    return samples


# ========== Main download function ==========

def prepare_large_qa_data(test_mode: bool = False, only_chinese: bool = False,
                          only_english: bool = False):
    """Download and merge large-scale QA datasets."""
    from datasets import load_dataset

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_all = []
    dev_all = []

    # Test mode limits
    max_train = 10000 if test_mode else 0
    max_dev = 2000 if test_mode else 0

    print("=" * 70)
    print(f"Preparing Large-Scale QA Data ({'TEST MODE' if test_mode else 'FULL MODE'})")
    print("=" * 70)
    print()

    # Language filter
    skip_chinese = only_english
    skip_english = only_chinese

    # ========== English Datasets ==========

    if not skip_english:
        # 1. SQuAD v1.1
        print("[1/9] Loading SQuAD v1.1 ...")
        try:
            sq1_train = _convert_squad(load_dataset("rajpurkar/squad", split="train"),
                                       max_train, source="squad_v1")
            sq1_dev = _convert_squad(load_dataset("rajpurkar/squad", split="validation"),
                                     max_dev, source="squad_v1")
            train_all.extend(sq1_train)
            dev_all.extend(sq1_dev)
            print(f"  ✓ SQuAD v1: train={len(sq1_train):,}, dev={len(sq1_dev):,}")
        except Exception as e:
            print(f"  ✗ SQuAD v1 failed: {e}")
            traceback.print_exc()

        # 2. SQuAD v2.0 (with unanswerable questions)
        print("[2/9] Loading SQuAD v2.0 ...")
        try:
            sq2_train = _convert_squad(load_dataset("rajpurkar/squad_v2", split="train"),
                                       max_train, source="squad_v2", include_unanswerable=True)
            sq2_dev = _convert_squad(load_dataset("rajpurkar/squad_v2", split="validation"),
                                     max_dev, source="squad_v2", include_unanswerable=True)
            train_all.extend(sq2_train)
            dev_all.extend(sq2_dev)
            print(f"  ✓ SQuAD v2: train={len(sq2_train):,}, dev={len(sq2_dev):,}")
        except Exception as e:
            print(f"  ✗ SQuAD v2 failed: {e}")
            traceback.print_exc()

        # 3. TriviaQA
        print("[3/9] Loading TriviaQA (streaming) ...")
        try:
            tqa_train = _convert_triviaqa(
                load_dataset("trivia_qa", "rc", split="train", streaming=True),
                max_train, source="triviaqa")
            tqa_dev = _convert_triviaqa(
                load_dataset("trivia_qa", "rc", split="validation", streaming=True),
                max_dev, source="triviaqa")
            train_all.extend(tqa_train)
            dev_all.extend(tqa_dev)
            print(f"  ✓ TriviaQA: train={len(tqa_train):,}, dev={len(tqa_dev):,}")
        except Exception as e:
            print(f"  ✗ TriviaQA failed: {e}")
            traceback.print_exc()

        # 4. Natural Questions (large!)
        if not test_mode:  # Skip in test mode due to size
            print("[4/9] Loading Natural Questions (large, may take time) ...")
            try:
                nq_train = _convert_natural_questions(
                    load_dataset("natural_questions", split="train", streaming=True),
                    max_n=150000,  # Limit to 150K to avoid excessive size
                    source="natural_questions")
                nq_dev = _convert_natural_questions(
                    load_dataset("natural_questions", split="validation", streaming=True),
                    max_dev, source="natural_questions")
                train_all.extend(nq_train)
                dev_all.extend(nq_dev)
                print(f"  ✓ Natural Questions: train={len(nq_train):,}, dev={len(nq_dev):,}")
            except Exception as e:
                print(f"  ✗ Natural Questions failed: {e}")
                traceback.print_exc()
        else:
            print("[4/9] Natural Questions: skipped in test mode")

        # 5. HotpotQA (multi-hop)
        print("[5/9] Loading HotpotQA ...")
        try:
            hpqa_train = _convert_hotpotqa(
                load_dataset("hotpot_qa", "distractor", split="train"),
                max_train, source="hotpotqa")
            hpqa_dev = _convert_hotpotqa(
                load_dataset("hotpot_qa", "distractor", split="validation"),
                max_dev, source="hotpotqa")
            train_all.extend(hpqa_train)
            dev_all.extend(hpqa_dev)
            print(f"  ✓ HotpotQA: train={len(hpqa_train):,}, dev={len(hpqa_dev):,}")
        except Exception as e:
            print(f"  ✗ HotpotQA failed: {e}")
            traceback.print_exc()
    else:
        print("[1-5/9] English datasets: skipped (only_chinese=True)")

    # ========== Chinese Datasets ==========

    if not skip_chinese:
        # 6. CMRC2018
        print("[6/9] Loading CMRC2018 ...")
        try:
            cm_train = _convert_cmrc_or_drcd(load_dataset("cmrc2018", split="train"),
                                             max_train, source="cmrc")
            cm_dev = _convert_cmrc_or_drcd(load_dataset("cmrc2018", split="validation"),
                                           max_dev, source="cmrc")
            train_all.extend(cm_train)
            dev_all.extend(cm_dev)
            print(f"  ✓ CMRC2018: train={len(cm_train):,}, dev={len(cm_dev):,}")
        except Exception as e:
            print(f"  ✗ CMRC2018 failed: {e}")
            traceback.print_exc()

        # 7. DuReader
        print("[7/9] Loading DuReader ...")
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
                    print(f"  ✓ DuReader: train={len(d_train):,}, dev={len(d_dev):,}")
                    loaded = True
                    break
                except Exception:
                    continue
            if not loaded:
                print(f"  ✗ DuReader: all repos failed")
        except Exception as e:
            print(f"  ✗ DuReader failed: {e}")
            traceback.print_exc()

        # 8. DRCD (Traditional Chinese)
        print("[8/9] Loading DRCD ...")
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
                    print(f"  ✓ DRCD: train={len(dr_train):,}, dev={len(dr_dev):,}")
                    drcd_loaded = True
                    break
                except Exception:
                    continue
            if not drcd_loaded:
                print(f"  ✗ DRCD: all repos failed")
        except Exception as e:
            print(f"  ✗ DRCD failed: {e}")
            traceback.print_exc()

        # 9. WebQA (Chinese web)
        print("[9/9] Loading WebQA ...")
        try:
            # Try multiple repo names
            webqa_loaded = False
            for repo in ["suolyer/webqa", "THUDM/webqa"]:
                try:
                    wq_train = _convert_webqa(load_dataset(repo, split="train"),
                                              max_train, source="webqa")
                    # WebQA may not have separate dev set
                    try:
                        wq_dev = _convert_webqa(load_dataset(repo, split="validation"),
                                                max_dev, source="webqa")
                    except:
                        wq_dev = []
                    train_all.extend(wq_train)
                    dev_all.extend(wq_dev)
                    print(f"  ✓ WebQA: train={len(wq_train):,}, dev={len(wq_dev):,}")
                    webqa_loaded = True
                    break
                except Exception:
                    continue
            if not webqa_loaded:
                print(f"  ✗ WebQA: all repos failed or not available")
        except Exception as e:
            print(f"  ✗ WebQA failed: {e}")
            traceback.print_exc()
    else:
        print("[6-9/9] Chinese datasets: skipped (only_english=True)")

    # ========== Shuffle and Save ==========

    print()
    print("=" * 70)
    print("Finalizing datasets...")
    print("=" * 70)

    random.seed(42)
    random.shuffle(train_all)
    random.shuffle(dev_all)

    # Save
    train_path = DATA_DIR / "qa_large_train.json"
    dev_path = DATA_DIR / "qa_large_dev.json"

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_all, f, ensure_ascii=False, indent=None)

    with open(dev_path, "w", encoding="utf-8") as f:
        json.dump(dev_all, f, ensure_ascii=False, indent=None)

    # Summary
    print(f"\n✓ Train: {len(train_all):,} samples → {train_path}")
    print(f"  Size: {train_path.stat().st_size / 1e6:.1f} MB")
    print(f"\n✓ Dev:   {len(dev_all):,} samples → {dev_path}")
    print(f"  Size: {dev_path.stat().st_size / 1e6:.1f} MB")

    # Source breakdown
    print("\n📊 Source Breakdown:")
    train_sources = {}
    for item in train_all:
        src = item.get("source", "unknown")
        train_sources[src] = train_sources.get(src, 0) + 1

    for src, count in sorted(train_sources.items()):
        pct = 100 * count / len(train_all)
        print(f"  {src:20s}: {count:7,} ({pct:5.1f}%)")

    print("\n" + "=" * 70)
    print("✓ Large-scale QA data preparation complete!")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare large-scale QA datasets for pure QA training"
    )
    parser.add_argument("--test", action="store_true",
                        help="Test mode: download small subsets (~10K train, ~2K dev)")
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
