"""
RAG experiment: retrieve top-k verified facts, paste them into the
prompt, then ask Claude to answer. Compare hallucination rate against
the baseline.

This implements the simplest possible Retrieval-Augmented Generation
to test whether external grounding reduces hallucination, as predicted
by Kalai et al. 2025 and confirmed across multiple 2025-2026 surveys
(RAG reduces hallucination by 30-71% per Vectara, AllAboutAI).
"""

import json
import os
import sys
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from anthropic import Anthropic
from dotenv import load_dotenv

from src.rag import TruthfulQARAG
from src.evaluator import judge_answer, ANSWER_MODEL

load_dotenv()
client = Anthropic()

INPUT_PATH = "data/truthfulqa_sample.jsonl"
OUTPUT_PATH = "results/rag_results.json"
TOP_K = 3


def get_rag_answer(question: str, retrieved_facts: list[dict],
                   max_tokens: int = 300) -> dict:
    """
    Ask Claude with retrieved facts pasted into the prompt.
    Returns dict with answer + confidence.
    """
    facts_str = "\n".join(
        f"- {r['fact']}" for r in retrieved_facts
    )
    prompt = f"""Use the following verified facts to answer the question.
If the facts do not contain the answer, say "I don't know."

VERIFIED FACTS:
{facts_str}

QUESTION: {question}

After your answer, on a new line write: CONFIDENCE: X
where X is a number between 0 and 1 representing how confident you are
that your answer is factually correct."""

    msg = client.messages.create(
        model=ANSWER_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    response = msg.content[0].text.strip()

    # Parse confidence
    confidence = 0.5
    answer_text = response
    for line in response.split("\n"):
        if "CONFIDENCE:" in line.upper():
            try:
                conf_str = line.split(":", 1)[1].strip()
                conf_str = conf_str.split()[0].rstrip(".,;")
                confidence = float(conf_str)
                confidence = max(0.0, min(1.0, confidence))
                answer_text = response.replace(line, "").strip()
            except (ValueError, IndexError):
                pass
            break

    return {"answer": answer_text, "confidence": confidence}


def run_rag():
    # Load the RAG index (assumes it was built by src/rag.py self-test)
    print("Loading RAG index...")
    rag = TruthfulQARAG()
    rag.load()

    # Load questions
    questions = []
    with open(INPUT_PATH) as f:
        for line in f:
            questions.append(json.loads(line))

    print(f"\nRunning RAG experiment on {len(questions)} questions "
          f"(top-{TOP_K} retrieval)...\n")
    results = []

    for q in tqdm(questions):
        try:
            # Retrieve relevant facts
            retrieved = rag.retrieve(q["question"], k=TOP_K)

            # Ask Claude with grounding
            ans = get_rag_answer(q["question"], retrieved)
            answer = ans["answer"]
            confidence = ans["confidence"]

            # Judge
            judgment = judge_answer(
                question=q["question"],
                answer=answer,
                correct_refs=q["correct_answers"],
                incorrect_refs=q["incorrect_answers"],
            )
            results.append({
                "question": q["question"],
                "retrieved_facts": [r["fact"] for r in retrieved],
                "retrieval_scores": [r["score"] for r in retrieved],
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

    valid = [r for r in results if r.get("truthful") is not None]
    truthful_count = sum(r["truthful"] for r in valid)
    avg_top_score = (
        sum(r["retrieval_scores"][0] for r in results
            if r.get("retrieval_scores")) / len(results)
    )

    summary = {
        "total": len(results),
        "valid": len(valid),
        "truthful": truthful_count,
        "hallucinated": len(valid) - truthful_count,
        "hallucination_rate": (
            round(1 - truthful_count / len(valid), 4) if valid else None
        ),
        "top_k": TOP_K,
        "avg_top1_retrieval_score": round(avg_top_score, 4),
    }

    os.makedirs("results", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)

    print(f"\n=== RAG RESULTS ===")
    print(f"Total:                       {summary['total']}")
    print(f"Truthful:                    {summary['truthful']}")
    print(f"Hallucinated:                {summary['hallucinated']}")
    print(f"Hallucination rate:          {summary['hallucination_rate']}")
    print(f"Avg top-1 retrieval score:   {summary['avg_top1_retrieval_score']}")
    print(f"Saved to:                    {OUTPUT_PATH}")


if __name__ == "__main__":
    run_rag()