# Mitigating Hallucination in LLMs through Data Normalization

A baseline pipeline for evaluating whether surface-level data normalization reduces hallucination rates in Large Language Models, using TruthfulQA as the benchmark and Claude Haiku 4.5 as the model under test.

This work was completed as part of the Research Assistant hiring task for the CANSSI-funded LLM project at Wilfrid Laurier University (Spring 2026), supervised by Prof. Sukhjit Singh Sehra.

## Research Question

Does normalizing input data (text, numbers, dates) before generation reduce hallucination rates compared to a no-normalization baseline?

## Methodology

- **Dataset:** TruthfulQA (validation split), 50-question sample
- **Model:** Claude Haiku 4.5 (`claude-haiku-4-5-20251001`)
- **Judge:** LLM-as-judge using Claude Haiku 4.5 with the dataset's correct/incorrect reference answers
- **Normalization strategies:**
  1. **Text normalization** — Unicode NFKC, lowercasing, whitespace collapse
  2. **Numerical & date normalization** — word-numbers to digits, percent unification, dates canonicalized to `YYYY-MM-DD`
- **Metric:** hallucination rate = % of answers judged not aligned with TruthfulQA's correct references

## Results

| Condition  | N  | Truthful | Hallucinated | Hallucination Rate |
|------------|----|----------|--------------|--------------------|
| Baseline   | 50 | 41       | 9            | 18.00%             |
| Normalized | 50 | 39       | 11           | 22.00%             |

**Absolute change: +4.00%** (normalization slightly increased hallucinations).

## Discussion

Normalization did not reduce hallucination on TruthfulQA. This is consistent with the dataset's design — TruthfulQA probes for misconceptions and common false beliefs, not formatting inconsistencies. Surface-level normalization changes how data *looks*, not the conceptual content that drives TruthfulQA hallucinations.

This finding aligns with Li et al. (2025), which argues that knowledge-based hallucinations require external knowledge grounding (RAG), not input preprocessing. It also suggests that data normalization may be more effective on benchmarks where formatting variance is the actual error source (e.g., RAGTruth, FEVER) — a direction for future work.

## Repository Structure

```
.
├── data/
│   ├── load_truthfulqa.py        # Load and sample TruthfulQA
│   └── truthfulqa_sample.jsonl   # 50-question sample
├── src/
│   ├── normalizers.py            # Two normalization strategies
│   └── evaluator.py              # Claude API + LLM-as-judge
├── experiments/
│   ├── run_baseline.py           # No-normalization baseline
│   ├── run_normalized.py         # With normalization applied
│   └── compare_results.py        # Generate comparison.md
├── results/
│   ├── baseline_results.json
│   ├── normalized_results.json
│   └── comparison.md
├── presentation.pptx             # Paper review presentation
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/ovxncdev/llm-hallucination-normalization.git
cd llm-hallucination-normalization
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
ANTHROPIC_API_KEY=sk-ant-your-key-here
```

## Usage

Run in this order:

```bash
# 1. Load dataset (one-time)
python3 data/load_truthfulqa.py

# 2. Run baseline (50 questions, ~3 min, ~$0.05)
python3 experiments/run_baseline.py

# 3. Run normalized version (~3 min, ~$0.05)
python3 experiments/run_normalized.py

# 4. Generate comparison
python3 experiments/compare_results.py
```

## Limitations

- **Small sample size (n=50).** The observed +4% difference is within noise.
- **Single judge model.** Using Claude to both answer and judge introduces possible bias; future work should use a separate judge model.
- **Limited normalization scope.** Only text and numerical/date normalization were tested. Entity normalization, evidence-chunk normalization, and embedding normalization remain unexplored.
- **Single benchmark.** TruthfulQA is misconception-focused; results may differ on RAGTruth or FEVER.

## Future Work

1. Larger sample (full 817 questions) and statistical significance testing (McNemar's test on paired outcomes).
2. Test normalization on **RAGTruth** and **FEVER**, where formatting consistency between evidence and claim plausibly matters more.
3. Add **entity normalization** (resolving aliases, e.g., "USA" / "U.S." / "United States").
4. Add **embedding-level normalization** (L2-normalized retrieval embeddings).
5. Replace LLM-as-judge with a fine-tuned classifier or human evaluation on a subset.

## References

- Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring How Models Mimic Human Falsehoods. *Proceedings of ACL 2022*, 3214–3252.
- Li, Y., Fu, X., Verma, G., Buitelaar, P., & Liu, M. (2025). Mitigating Hallucination in Large Language Models (LLMs): An Application-Oriented Survey on RAG, Reasoning, and Agentic Systems. *arXiv:2510.24476*.
- Wang, Z., Wang, J., Lu, Z., & You, F. (2026). Large language modeling of hallucinatory problem mitigation based on the wheel of emotions. *Neural Networks*, 193, 107996.

## License

MIT
