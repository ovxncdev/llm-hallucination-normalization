"""
Load TruthfulQA dataset and save a small sample for experiments.
"""

from datasets import load_dataset
import json
import os

OUTPUT_PATH = "data/truthfulqa_sample.jsonl"
SAMPLE_SIZE = 100  # Keep small for fast iteration and low API cost


def load_and_sample():
    print("Loading TruthfulQA from Hugging Face...")
    ds = load_dataset("truthful_qa", "generation", split="validation")
    print(f"Total questions in dataset: {len(ds)}")

    # Take first N questions
    sample = ds.select(range(SAMPLE_SIZE))

    os.makedirs("data", exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        for item in sample:
            record = {
                "question": item["question"],
                "best_answer": item["best_answer"],
                "correct_answers": item["correct_answers"],
                "incorrect_answers": item["incorrect_answers"],
                "category": item.get("category", ""),
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved {SAMPLE_SIZE} questions to {OUTPUT_PATH}")


if __name__ == "__main__":
    load_and_sample()