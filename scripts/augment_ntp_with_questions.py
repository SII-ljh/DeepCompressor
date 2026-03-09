"""Augment NTP data with pseudo-questions for question-guided compression.

Generates questions using three strategies:
1. Sample from QA dataset (70%) - Randomly select relevant questions
2. Extract key sentences (20%) - Use TF-IDF to find salient content
3. Generic questions (10%) - Fallback templates

Usage:
    python scripts/augment_ntp_with_questions.py \
        --input data/ntp_train.jsonl \
        --output data/ntp_train_guided.jsonl \
        --qa_source data/qa_train.json \
        --strategy mixed
"""

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Generic question templates (Chinese financial domain)
GENERIC_QUESTIONS = [
    "请总结这篇文章的主要内容。",
    "这份文档的核心信息是什么?",
    "请概括文中的关键要点。",
    "这篇报告讨论了哪些重要内容?",
    "文档中提到了哪些关键数据?",
    "请提取文中的主要财务信息。",
    "这份公告的重点内容有哪些?",
    "文中涉及哪些重要事项?",
    "请归纳文档的主要观点。",
    "这份材料的关键信息是什么?",
]


def load_qa_questions(qa_path: str) -> List[str]:
    """Load all questions from QA dataset."""
    with open(qa_path) as f:
        qa_data = json.load(f)
    return [item["question"] for item in qa_data]


def extract_key_sentence_question(text: str, vectorizer: TfidfVectorizer = None) -> str:
    """Extract a key sentence and convert to question format.

    Uses TF-IDF to find the most salient sentence, then wraps it
    in a question template.
    """
    # Split into sentences (simple approach for Chinese)
    sentences = [s.strip() for s in text.split("。") if len(s.strip()) > 10]

    if not sentences:
        return random.choice(GENERIC_QUESTIONS)

    if len(sentences) == 1:
        # Only one sentence, wrap it directly
        topic = sentences[0][:20]  # First 20 chars
        return f"关于「{topic}」的内容是什么?"

    # Use TF-IDF to find most salient sentence
    try:
        if vectorizer is None:
            vectorizer = TfidfVectorizer(max_features=100)

        tfidf_matrix = vectorizer.fit_transform(sentences)
        sentence_scores = tfidf_matrix.sum(axis=1).A1  # Sum of TF-IDF scores per sentence
        best_idx = sentence_scores.argmax()

        key_sentence = sentences[best_idx]
        # Extract topic (first 15-20 characters)
        topic = key_sentence[:20] if len(key_sentence) > 20 else key_sentence

        return f"文中关于「{topic}」说了什么?"
    except Exception:
        # Fallback to generic if TF-IDF fails
        return random.choice(GENERIC_QUESTIONS)


def augment_with_questions(
    input_path: str,
    output_path: str,
    qa_source: str = None,
    strategy: str = "mixed",
    seed: int = 42,
):
    """Augment NTP data with questions.

    Args:
        input_path: Path to input NTP JSONL file
        output_path: Path to output augmented JSONL file
        qa_source: Path to QA JSON file (for sampling questions)
        strategy: "mixed" (70/20/10), "qa_only", "extract_only", "generic_only"
        seed: Random seed
    """
    random.seed(seed)

    # Load QA questions if available
    qa_questions = []
    if qa_source and Path(qa_source).exists():
        qa_questions = load_qa_questions(qa_source)
        logger.info(f"Loaded {len(qa_questions):,} questions from {qa_source}")

    # Initialize TF-IDF vectorizer for extraction strategy
    vectorizer = TfidfVectorizer(max_features=100)

    # Process input file
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats = {"qa_sample": 0, "extract": 0, "generic": 0, "total": 0}

    with open(input_path, "r") as fin, open(output_path, "w") as fout:
        for line_num, line in enumerate(fin, 1):
            if not line.strip():
                continue

            try:
                data = json.loads(line)
                text = data["text"]

                # Choose question generation strategy
                if strategy == "mixed":
                    roll = random.random()
                    if roll < 0.7 and qa_questions:
                        # 70% sample from QA
                        question = random.choice(qa_questions)
                        stats["qa_sample"] += 1
                    elif roll < 0.9:
                        # 20% extract key sentence
                        question = extract_key_sentence_question(text, vectorizer)
                        stats["extract"] += 1
                    else:
                        # 10% generic
                        question = random.choice(GENERIC_QUESTIONS)
                        stats["generic"] += 1
                elif strategy == "qa_only":
                    if qa_questions:
                        question = random.choice(qa_questions)
                        stats["qa_sample"] += 1
                    else:
                        question = random.choice(GENERIC_QUESTIONS)
                        stats["generic"] += 1
                elif strategy == "extract_only":
                    question = extract_key_sentence_question(text, vectorizer)
                    stats["extract"] += 1
                elif strategy == "generic_only":
                    question = random.choice(GENERIC_QUESTIONS)
                    stats["generic"] += 1
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")

                # Add question to data
                data["question"] = question
                fout.write(json.dumps(data, ensure_ascii=False) + "\n")
                stats["total"] += 1

                if line_num % 100000 == 0:
                    logger.info(f"Processed {line_num:,} lines...")

            except Exception as e:
                logger.warning(f"Failed to process line {line_num}: {e}")
                continue

    logger.info(f"✓ Augmented {stats['total']:,} samples → {output_path}")
    logger.info(f"  Strategy breakdown:")
    logger.info(f"    QA sampled: {stats['qa_sample']:,} ({100*stats['qa_sample']/stats['total']:.1f}%)")
    logger.info(f"    Extracted:  {stats['extract']:,} ({100*stats['extract']/stats['total']:.1f}%)")
    logger.info(f"    Generic:    {stats['generic']:,} ({100*stats['generic']/stats['total']:.1f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Augment NTP data with pseudo-questions"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="Input NTP JSONL file")
    parser.add_argument("--output", type=str, required=True,
                        help="Output augmented JSONL file")
    parser.add_argument("--qa_source", type=str, default=None,
                        help="QA JSON file for sampling questions")
    parser.add_argument("--strategy", type=str, default="mixed",
                        choices=["mixed", "qa_only", "extract_only", "generic_only"],
                        help="Question generation strategy")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")

    args = parser.parse_args()

    augment_with_questions(
        input_path=args.input,
        output_path=args.output,
        qa_source=args.qa_source,
        strategy=args.strategy,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
