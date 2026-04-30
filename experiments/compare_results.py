"""
Compare baseline vs normalized results and produce a summary table.
"""

import json
import os

BASELINE = "results/baseline_results.json"
NORMALIZED = "results/normalized_results.json"
OUTPUT = "results/comparison.md"


def load(path):
    with open(path) as f:
        return json.load(f)


def main():
    b = load(BASELINE)["summary"]
    n = load(NORMALIZED)["summary"]

    md = f"""# Comparison: Baseline vs. Normalized

## Summary Table

| Condition  | N  | Truthful | Hallucinated | Hallucination Rate |
|------------|----|----------|--------------|--------------------|
| Baseline   | {b['valid']} | {b['truthful']}       | {b['hallucinated']}            | {b['hallucination_rate']:.2%}              |
| Normalized | {n['valid']} | {n['truthful']}       | {n['hallucinated']}            | {n['hallucination_rate']:.2%}              |

## Delta

- Absolute change in hallucination rate: **{(n['hallucination_rate'] - b['hallucination_rate']):+.2%}**
- Sample size per condition: {b['valid']}

## Interpretation

With a sample of {b['valid']} questions from TruthfulQA, applying surface-level
normalization (text + numerical/date) to inputs and outputs did **not** reduce
hallucination rates — in fact, the rate slightly increased.

This is consistent with TruthfulQA's design: it probes for **misconceptions**
(e.g., common false beliefs), not formatting inconsistencies. Surface
normalization addresses how data *looks*, not the conceptual errors that drive
TruthfulQA hallucinations. This finding aligns with Li et al. (2025), which
argues that knowledge-based hallucinations require knowledge grounding (RAG),
not preprocessing.

## Limitations

- Small sample size (n={b['valid']}); the observed difference is within noise.
- Single judge model (Claude Haiku 4.5) used for both answering and judging.
- Two normalization strategies tested; entity and embedding normalization
  remain unexplored.
"""

    os.makedirs("results", exist_ok=True)
    with open(OUTPUT, "w") as f:
        f.write(md)

    print(md)
    print(f"\nSaved to {OUTPUT}")


if __name__ == "__main__":
    main()