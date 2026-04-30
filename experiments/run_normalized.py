"""
Normalized experiment: same as baseline, but apply normalization
to both the question (before sending to Claude) and the answer
(before sending to the judge).
"""

import json
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluator import get_answer, judge_answer
from src.normalizers import normalize_text, normalize_numbers_and_dates

INPUT_PATH = "data/truthfulqa_sample.jsonl"
OUTPUT_PATH = "results/normalized_results.json"


def normalize(s: str) -> str:
    """Apply both normalization strategies."""
    s = normalize_text(s)
    s = normalize_numbers_and_dates(s)
    return s


def run_normalized():
    questions = []
    with open(INPUT_PATH) as f:
        for line in f:
            questions.append(json.loads(line))

    print(f"Running NORMALIZED experiment on {len(questions)} questions...")
    results = []

    for q in tqdm(questions):
        try:
            # Normalize question before asking Claude
            norm_question = normalize(q["question"])
            answer = get_answer(norm_question)
            # Normalize the answer before judging
            norm_answer = normalize(answer)
            # Normalize references too, so the judge compares like-to-like
            norm_correct = [normalize(r) for r in q["correct_answers"]]
            norm_incorrect = [normalize(r) for r in q["incorrect_answers"]]

            judgment = judge_answer(
                question=norm_question,
                answer=norm_answer,
                correct_refs=norm_correct,
                incorrect_refs=norm_incorrect,
            )
            results.append({
                "original_question": q["question"],
                "normalized_question": norm_question,
                "answer": answer,
                "normalized_answer": norm_answer,
                "truthful": judgment["truthful"],
                "reasoning": judgment["reasoning"],
                "category": q["category"],
            })
        except Exception as e:
            print(f"\nError on question: {q['question'][:60]}... -> {e}")
            results.append({
                "original_question": q["question"],
                "answer": None,
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
    }

    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\n=== NORMALIZED RESULTS ===")
    print(f"Total:              {summary['total']}")
    print(f"Truthful:           {summary['truthful']}")
    print(f"Hallucinated:       {summary['hallucinated']}")
    print(f"Hallucination rate: {summary['hallucination_rate']}")
    print(f"Saved to:           {OUTPUT_PATH}")


if __name__ == "__main__":
    run_normalized()