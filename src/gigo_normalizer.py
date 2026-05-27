"""
GIGO (Garbage-In-Garbage-Out) Normalization.

Following Kalai et al. 2025 Section 3.4, which frames data normalization
as a GIGO intervention. This module implements contradiction-aware
normalization: when the same underlying fact appears with conflicting
surface forms, collapse to a canonical representation.

Approach:
1. For each reference answer set, find near-duplicate facts using
   simple token-set similarity (Jaccard).
2. When near-duplicates exist, keep the most common surface form
   (or the longest one, as a tiebreaker, since longer answers tend
   to be more precise).
3. This reduces the effective monofact rate.
"""

import json
import re
from collections import Counter


def tokenize(s: str) -> set:
    """Crude tokenizer for similarity computation."""
    return set(re.findall(r"\b\w+\b", s.lower()))


def jaccard_similarity(a: str, b: str) -> float:
    """Jaccard similarity between two strings on token sets."""
    ta, tb = tokenize(a), tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def find_clusters(facts: list[str], threshold: float = 0.6) -> list[list[int]]:
    """
    Group facts into clusters where each pair has Jaccard similarity
    above threshold. Returns list of clusters (each cluster is a list
    of indices into facts).
    """
    n = len(facts)
    assigned = [-1] * n
    clusters = []

    for i in range(n):
        if assigned[i] != -1:
            continue
        cluster = [i]
        assigned[i] = len(clusters)
        for j in range(i + 1, n):
            if assigned[j] != -1:
                continue
            if jaccard_similarity(facts[i], facts[j]) >= threshold:
                cluster.append(j)
                assigned[j] = len(clusters)
        clusters.append(cluster)

    return clusters


def canonicalize_facts(facts: list[str], threshold: float = 0.6) -> list[str]:
    """
    Apply GIGO normalization: collapse near-duplicate facts to a single
    canonical form (the longest/most informative).
    """
    if not facts:
        return facts

    clusters = find_clusters(facts, threshold)
    canonical = []
    for cluster in clusters:
        # Pick the longest fact in each cluster as the canonical form
        members = [facts[i] for i in cluster]
        canonical.append(max(members, key=len))
    return canonical


def normalize_reference_set(item: dict, threshold: float = 0.6) -> dict:
    """
    Apply GIGO normalization to a single TruthfulQA item's reference
    answers (correct_answers and incorrect_answers).
    """
    result = dict(item)
    result["correct_answers"] = canonicalize_facts(
        item.get("correct_answers", []), threshold
    )
    result["incorrect_answers"] = canonicalize_facts(
        item.get("incorrect_answers", []), threshold
    )
    return result


# Self-test
if __name__ == "__main__":
    print("=== GIGO Normalization Self-Test ===\n")

    sample_facts = [
        "The Eiffel Tower is in Paris.",
        "The Eiffel Tower is located in Paris, France.",  # near-duplicate
        "Paris contains the Eiffel Tower.",  # different but similar
        "The Great Wall is in China.",  # different fact
    ]

    print("Original facts:")
    for f in sample_facts:
        print(f"  - {f}")
    print()

    clusters = find_clusters(sample_facts, threshold=0.4)
    print(f"Clusters (threshold=0.4): {clusters}")
    print()

    canonical = canonicalize_facts(sample_facts, threshold=0.4)
    print(f"Canonical forms ({len(canonical)} kept):")
    for c in canonical:
        print(f"  - {c}")
    print()

    # Show reduction on TruthfulQA
    print("=== Effect on TruthfulQA sample ===\n")
    with open("data/truthfulqa_sample.jsonl") as f:
        items = [json.loads(line) for line in f]

    total_correct_before = sum(len(it["correct_answers"]) for it in items)
    total_incorrect_before = sum(len(it["incorrect_answers"]) for it in items)

    normalized = [normalize_reference_set(it, threshold=0.6) for it in items]
    total_correct_after = sum(len(it["correct_answers"]) for it in normalized)
    total_incorrect_after = sum(
        len(it["incorrect_answers"]) for it in normalized
    )

    print(
        f"Correct refs:   {total_correct_before} -> {total_correct_after} "
        f"(Δ = {total_correct_after - total_correct_before})"
    )
    print(
        f"Incorrect refs: {total_incorrect_before} -> {total_incorrect_after} "
        f"(Δ = {total_incorrect_after - total_incorrect_before})"
    )