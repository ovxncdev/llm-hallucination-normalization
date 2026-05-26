"""
Baseline experiment: ask Claude TruthfulQA questions WITHOUT normalization.
Save raw answers and judge scores to a JSON file.
"""

import json
import os
import sys
from tqdm import tqdm

# Allow imports from src/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluator import get_answer_with_confidence, judge_answer

INPUT_PATH = "data/truthfulqa_sample.jsonl"
OUTPUT_PATH = "results/baseline_results.json"


def run_baseline():
    # Load questions
    questions = []
    with open(INPUT_PATH) as f:
        for line in f:
            questions.append(json.loads(line))

    print(f"Running baseline on {len(questions)} questions...")
    results = []

    for q in tqdm(questions):
        try:
            result = get_answer_with_confidence(q["question"])
            answer = result["answer"]
            confidence = result["confidence"]
            judgment = judge_answer(
                question=q["question"],
                answer=answer,
                correct_refs=q["correct_answers"],
                incorrect_refs=q["incorrect_answers"],
            )
            results.append({
                "question": q["question"],
                "answer": answer,
                "confidence": confidence,
                "truthful": judgment["truthful"],
                "reasoning": judgment["reasoning"],
                "category": q["category"],
            })
        except Exception as e:
            print(f"\nError on question: {q['question'][:60]}... -> {e}")
            results.append({
                "question": q["question"],
                "answer": None,
                "truthful": None,
                "error": str(e),
                "category": q["category"],
            })

    # Compute summary
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

    print(f"\n=== BASELINE RESULTS ===")
    print(f"Total:              {summary['total']}")
    print(f"Truthful:           {summary['truthful']}")
    print(f"Hallucinated:       {summary['hallucinated']}")
    print(f"Hallucination rate: {summary['hallucination_rate']}")
    print(f"Saved to:           {OUTPUT_PATH}")


if __name__ == "__main__":
    run_baseline()