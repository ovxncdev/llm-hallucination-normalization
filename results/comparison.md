# Comparison: Baseline vs. Normalized

## Summary Table

| Condition  | N  | Truthful | Hallucinated | Hallucination Rate |
|------------|----|----------|--------------|--------------------|
| Baseline   | 50 | 41       | 9            | 18.00%              |
| Normalized | 50 | 39       | 11            | 22.00%              |

## Delta

- Absolute change in hallucination rate: **+4.00%**
- Sample size per condition: 50

## Interpretation

With a sample of 50 questions from TruthfulQA, applying surface-level
normalization (text + numerical/date) to inputs and outputs did **not** reduce
hallucination rates — in fact, the rate slightly increased.

This is consistent with TruthfulQA's design: it probes for **misconceptions**
(e.g., common false beliefs), not formatting inconsistencies. Surface
normalization addresses how data *looks*, not the conceptual errors that drive
TruthfulQA hallucinations. This finding aligns with Li et al. (2025), which
argues that knowledge-based hallucinations require knowledge grounding (RAG),
not preprocessing.

## Limitations

- Small sample size (n=50); the observed difference is within noise.
- Single judge model (Claude Haiku 4.5) used for both answering and judging.
- Two normalization strategies tested; entity and embedding normalization
  remain unexplored.
