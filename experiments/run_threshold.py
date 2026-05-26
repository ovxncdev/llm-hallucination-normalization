"""
Confidence-thresholded experiment (Kalai et al. 2025 mitigation):
Instruct Claude to abstain ("I don't know") when below a confidence
threshold. Measure:
- Hallucination rate on ANSWERED questions (the key metric)
- Abstention rate (how often the model said IDK)
- Overall accuracy (correct answers / total questions)

This implements the explicit mitigation proposed in
"Why Language Models Hallucinate" (Kalai, Nachum, Vempala, Zhang 2025).
"""

import json
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluator import get_answer_with_threshold, judge_answer

INPUT_PATH = "data/truthfulqa_sample.jsonl"
THRESHOLD = 0.9  # Following Kalai et al. 2025; try 0.5, 0.75, 0.9
OUTPUT_PATH = f"results/threshold_{int(THRESHOLD*100)}_results.json"


def run_threshold_experiment():
    questions = []
    with open(INPUT_PATH) as f:
        for line in f:
            questions.append(json.loads(line))

    print(f"Running CONFIDENCE-THRESHOLDED experiment (t={THRESHOLD}) on "
          f"{len(questions)} questions...")
    results = []

    for q in tqdm(questions):
        try:
            response = get_answer_with_threshold(
                q["question"], threshold=THRESHOLD
            )
            answer = response["answer"]
            confidence = response["confidence"]
            abstained = response["abstained"]

            if abstained:
                # No need to judge — abstention is not a hallucination
                results.append({
                    "question": q["question"],
                    "answer": answer,
                    "confidence": confidence,
                    "abstained": True,
                    "truthful": None,  # not applicable
                    "category": q["category"],
                })
            else:
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
                    "abstained": False,
                    "truthful": judgment["truthful"],
                    "reasoning": judgment["reasoning"],
                    "category": q["category"],
                })
        except Exception as e:
            print(f"\nError on question: {q['question'][:60]}... -> {e}")
            results.append({
                "question": q["question"],
                "answer": None,
                "confidence": None,
                "abstained": None,
                "truthful": None,
                "error": str(e),
                "category": q["category"],
            })

    # Summary statistics
    total = len(results)
    abstained_count = sum(1 for r in results if r.get("abstained") is True)
    answered = [r for r in results if r.get("abstained") is False
                and r.get("truthful") is not None]
    answered_count = len(answered)
    truthful_count = sum(r["truthful"] for r in answered)
    hallucinated_count = answered_count - truthful_count

    summary = {
        "threshold": THRESHOLD,
        "total": total,
        "abstained": abstained_count,
        "answered": answered_count,
        "truthful": truthful_count,
        "hallucinated": hallucinated_count,
        "abstention_rate": round(abstained_count / total, 4) if total else 0.0,
        "hallucination_rate_on_answered": (
            round(hallucinated_count / answered_count, 4)
            if answered_count else None
        ),
        "overall_accuracy": (
            round(truthful_count / total, 4) if total else 0.0
        ),
    }

    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\n=== THRESHOLDED RESULTS (t={THRESHOLD}) ===")
    print(f"Total questions:                  {summary['total']}")
    print(f"Abstained (said IDK):             {summary['abstained']}")
    print(f"Answered:                         {summary['answered']}")
    print(f"  - Truthful:                     {summary['truthful']}")
    print(f"  - Hallucinated:                 {summary['hallucinated']}")
    print(f"Abstention rate:                  {summary['abstention_rate']:.2%}")
    print(f"Hallucination rate (on answered): "
          f"{summary['hallucination_rate_on_answered']}")
    print(f"Overall accuracy:                 {summary['overall_accuracy']:.2%}")
    print(f"Saved to:                         {OUTPUT_PATH}")


if __name__ == "__main__":
    run_threshold_experiment()