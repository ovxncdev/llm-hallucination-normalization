"""
GIGO-normalized experiment: apply contradiction-aware normalization
to the reference answer sets before the judge evaluates Claude's answer.

This tests the prediction (Kalai et al. 2025 Section 3.4) that GIGO-
targeted normalization should reduce hallucination on benchmarks where
the reference data contains near-duplicate facts.
"""

import json
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluator import get_answer_with_confidence, judge_answer
from src.gigo_normalizer import canonicalize_facts

INPUT_PATH = "data/truthfulqa_sample.jsonl"
OUTPUT_PATH = "results/gigo_results.json"
SIMILARITY_THRESHOLD = 0.6


def run_gigo():
    questions = []
    with open(INPUT_PATH) as f:
        for line in f:
            questions.append(json.loads(line))

    print(f"Running GIGO-NORMALIZED experiment on {len(questions)} questions...")
    print(f"Jaccard similarity threshold: {SIMILARITY_THRESHOLD}\n")

    results = []
    total_collapsed_correct = 0
    total_collapsed_incorrect = 0

    for q in tqdm(questions):
        try:
            # Apply GIGO normalization to reference sets
            orig_correct = q["correct_answers"]
            orig_incorrect = q["incorrect_answers"]
            gigo_correct = canonicalize_facts(orig_correct, SIMILARITY_THRESHOLD)
            gigo_incorrect = canonicalize_facts(orig_incorrect, SIMILARITY_THRESHOLD)

            total_collapsed_correct += len(orig_correct) - len(gigo_correct)
            total_collapsed_incorrect += len(orig_incorrect) - len(gigo_incorrect)

            # Get Claude's answer (no normalization on the question itself)
            result = get_answer_with_confidence(q["question"])
            answer = result["answer"]
            confidence = result["confidence"]

            # Judge with the GIGO-normalized reference set
            judgment = judge_answer(
                question=q["question"],
                answer=answer,
                correct_refs=gigo_correct,
                incorrect_refs=gigo_incorrect,
            )
            results.append({
                "question": q["question"],
                "answer": answer,
                "confidence": confidence,
                "truthful": judgment["truthful"],
                "reasoning": judgment["reasoning"],
                "category": q["category"],
                "n_correct_refs_before": len(orig_correct),
                "n_correct_refs_after": len(gigo_correct),
                "n_incorrect_refs_before": len(orig_incorrect),
                "n_incorrect_refs_after": len(gigo_incorrect),
            })
        except Exception as e:
            print(f"\nError on question: {q['question'][:60]}... -> {e}")
            results.append({
                "question": q["question"],
                "answer": None,
                "confidence": None,
                "truthful": None,
                "error": str(e),
                "category": q["category"],
            })

    valid = [r for r in results if r.get("truthful") is not None]
    truthful_count = sum(r["truthful"] for r in valid)
    summary = {
        "total": len(results),
        "valid": len(valid),
        "truthful": truthful_count,
        "hallucinated": len(valid) - truthful_count,
        "hallucination_rate": (
            round(1 - truthful_count / len(valid), 4) if valid else None
        ),
        "collapsed_correct_refs": total_collapsed_correct,
        "collapsed_incorrect_refs": total_collapsed_incorrect,
        "similarity_threshold": SIMILARITY_THRESHOLD,
    }

    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\n=== GIGO-NORMALIZED RESULTS ===")
    print(f"Total:                       {summary['total']}")
    print(f"Truthful:                    {summary['truthful']}")
    print(f"Hallucinated:                {summary['hallucinated']}")
    print(f"Hallucination rate:          {summary['hallucination_rate']}")
    print(f"Collapsed correct refs:      {summary['collapsed_correct_refs']}")
    print(f"Collapsed incorrect refs:    {summary['collapsed_incorrect_refs']}")
    print(f"Saved to:                    {OUTPUT_PATH}")


if __name__ == "__main__":
    run_gigo()