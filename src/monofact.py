"""
MonoFact estimator (Good-Turing) for hallucination analysis.

Based on Kalai & Vempala (2024): "Calibrated Language Models Must Hallucinate".
The monofact rate is the fraction of facts that appear exactly once in the
training/reference data. It provides a statistical lower bound on the
hallucination rate for calibrated language models.

For TruthfulQA, we treat each question's "correct_answers" as the set of
canonical fact descriptions, and count how many distinct facts appear only
once across the entire dataset's reference answers.
"""

import json
from collections import Counter
from src.normalizers import normalize_text


def extract_facts(dataset_path: str, normalize: bool = False) -> list[str]:
    """Extract all reference facts from a TruthfulQA-style JSONL dataset."""
    facts = []
    with open(dataset_path) as f:
        for line in f:
            item = json.loads(line)
            for answer in item.get("correct_answers", []):
                if normalize:
                    answer = normalize_text(answer)
                facts.append(answer)
    return facts


def monofact_rate(facts: list[str]) -> dict:
    """
    Compute the monofact rate: fraction of distinct facts appearing exactly once.

    Returns dict with:
      - total_facts: total fact occurrences
      - unique_facts: number of distinct facts
      - monofacts: number of facts appearing exactly once
      - monofact_rate: monofacts / total_facts (the Good-Turing estimate)
      - unique_rate: unique_facts / total_facts
    """
    counts = Counter(facts)
    total = len(facts)
    unique = len(counts)
    monofacts = sum(1 for c in counts.values() if c == 1)

    return {
        "total_facts": total,
        "unique_facts": unique,
        "monofacts": monofacts,
        "monofact_rate": round(monofacts / total, 4) if total else 0.0,
        "unique_rate": round(unique / total, 4) if total else 0.0,
    }


if __name__ == "__main__":
    print("=== MonoFact analysis on TruthfulQA sample ===\n")

    # Without normalization
    raw_facts = extract_facts("data/truthfulqa_sample.jsonl", normalize=False)
    raw_stats = monofact_rate(raw_facts)
    print("Without normalization:")
    for k, v in raw_stats.items():
        print(f"  {k}: {v}")
    print()

    # With normalization
    norm_facts = extract_facts("data/truthfulqa_sample.jsonl", normalize=True)
    norm_stats = monofact_rate(norm_facts)
    print("With text normalization:")
    for k, v in norm_stats.items():
        print(f"  {k}: {v}")
    print()

    # Compare
    delta = norm_stats["monofact_rate"] - raw_stats["monofact_rate"]
    print(f"Δ monofact rate (normalized - raw): {delta:+.4f}")
    print(
        "Interpretation: a NEGATIVE delta means normalization is collapsing "
        "duplicate facts, lowering the monofact rate and the theoretical "
        "hallucination floor (per Kalai & Vempala 2024)."
    )