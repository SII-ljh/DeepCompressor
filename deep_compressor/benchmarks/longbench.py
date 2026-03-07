"""LongBench data adapter for Deep Compressor evaluation.

Loads QA subsets from HuggingFace datasets and converts them to the
internal {context, question, answer, answers, source} format used by
BenchmarkQADataset.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Core QA subsets commonly used by compression papers
# (ICAE, AutoCompressor, Gisting, etc.)
LONGBENCH_QA_SUBSETS: Dict[str, Dict] = {
    "multifieldqa_zh": {
        "dataset": "THUDM/LongBench",
        "subset": "multifieldqa_zh",
        "description": "Multi-domain single-doc QA (Chinese)",
    },
    "multifieldqa_en": {
        "dataset": "THUDM/LongBench",
        "subset": "multifieldqa_en",
        "description": "Multi-domain single-doc QA (English)",
    },
    "hotpotqa": {
        "dataset": "THUDM/LongBench",
        "subset": "hotpotqa",
        "description": "Multi-document QA (English, most cited)",
    },
    "narrativeqa": {
        "dataset": "THUDM/LongBench",
        "subset": "narrativeqa",
        "description": "Long narrative comprehension (English)",
    },
}


def _parse_answers(raw_answers) -> List[str]:
    """Parse LongBench answers field into a list of strings.

    LongBench stores answers as either a JSON-encoded list string or a plain
    string. Handle both cases.
    """
    if isinstance(raw_answers, list):
        return [str(a) for a in raw_answers]
    if isinstance(raw_answers, str):
        # Try JSON parse first (e.g. '["ans1", "ans2"]')
        try:
            parsed = json.loads(raw_answers)
            if isinstance(parsed, list):
                return [str(a) for a in parsed]
        except (json.JSONDecodeError, TypeError):
            pass
        return [raw_answers]
    return [str(raw_answers)]


def load_longbench_subset(subset_name: str,
                          cache_dir: Optional[str] = None) -> List[Dict]:
    """Load a LongBench QA subset from HuggingFace and convert to internal format.

    Args:
        subset_name: Key in LONGBENCH_QA_SUBSETS (e.g. "hotpotqa").
        cache_dir: Optional HuggingFace cache directory.

    Returns:
        List of dicts with keys: context, question, answer, answers, source.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "LongBench loading requires the `datasets` package. "
            "Install with: pip install datasets"
        )

    if subset_name not in LONGBENCH_QA_SUBSETS:
        logger.warning(
            f"Subset '{subset_name}' not in LONGBENCH_QA_SUBSETS. "
            f"Available: {list(LONGBENCH_QA_SUBSETS.keys())}. "
            f"Attempting to load anyway."
        )

    info = LONGBENCH_QA_SUBSETS.get(subset_name, {})
    dataset_name = info.get("dataset", "THUDM/LongBench")

    logger.info(f"Loading LongBench subset: {subset_name}")
    ds = load_dataset(dataset_name, subset_name, split="test",
                      cache_dir=cache_dir, trust_remote_code=True)

    records = []
    for item in ds:
        answers = _parse_answers(item.get("answers", item.get("answer", "")))
        records.append({
            "context": item["context"],
            "question": item.get("input", ""),
            "answer": answers[0] if answers else "",
            "answers": answers,
            "source": subset_name,
        })

    logger.info(f"Loaded {len(records)} samples from {subset_name}")
    return records


def save_as_internal_format(records: List[Dict], output_path: str) -> None:
    """Save records to local JSON for offline evaluation."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(records)} records to {output_path}")


def load_from_local(path: str) -> List[Dict]:
    """Load records from a local JSON file."""
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    logger.info(f"Loaded {len(records)} records from {path}")
    return records
