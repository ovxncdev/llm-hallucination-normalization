"""
Compute the Kalai-Vempala (2024) hallucination lower bound on
our empirical data, and compare to the observed hallucination rate.

Bound: g(H) >= MFd - Misb(g, p) - 3e^(-s)/delta - sqrt(6 ln(6/delta) / n)

where:
  MFd     = monofact rate (Good-Turing estimator)
  Misb    = miscalibration error (we use ECE as a proxy)
  s       = sparsity (log of |Hallucinations|/|Facts|)
  delta   = confidence level (we use 0.01 for 99%)
  n       = sample size
"""

import json
import math
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.monofact import extract_facts, monofact_rate
from src.calibration import compute_from_results_file


def kalai_vempala_bound(mfd: float, ece: float, n: int,
                       s: float = 10.0, delta: float = 0.01) -> dict:
    """
    Compute the Kalai-Vempala lower bound on hallucination rate.

    Args:
        mfd: monofact rate (Good-Turing estimate)
        ece: miscalibration error (proxy for Misb)
        n: sample size
        s: sparsity parameter (log ratio of hallucinations to facts);
           s=10 is generous, meaning |H| >= e^10 * |F| ~= 22,000x more
           hallucinations than facts (reasonable for open-domain QA)
        delta: confidence level (0.01 = 99% confidence)

    Returns:
        dict with bound, components, and interpretation
    """
    term_sparsity = (3 * math.exp(-s)) / delta
    term_sample = math.sqrt(6 * math.log(6 / delta) / n)
    bound = mfd - ece - term_sparsity - term_sample

    return {
        "mfd": mfd,
        "ece": ece,
        "n": n,
        "sparsity_s": s,
        "delta": delta,
        "term_mfd": mfd,
        "term_ece": -ece,
        "term_sparsity": -term_sparsity,
        "term_sample": -term_sample,
        "bound": max(0.0, bound),  # clamp to [0, 1]
        "bound_raw": bound,
    }


def main():
    print("=" * 60)
    print("Kalai-Vempala Hallucination Bound — Empirical Check")
    print("=" * 60)
    print()

    # Compute monofact rate
    facts = extract_facts("data/truthfulqa_sample.jsonl", normalize=False)
    mf = monofact_rate(facts)
    mfd = mf["monofact_rate"]
    n = mf["total_facts"]
    print(f"MonoFact rate (MFd):         {mfd:.4f}")
    print(f"Number of facts (n):         {n}")

    # Load ECE from baseline results
    ece_data = compute_from_results_file("results/baseline_results.json")
    ece = ece_data["ece"]
    print(f"Calibration error (ECE):     {ece:.4f}")
    print()

    # Compute the bound
    print("Theoretical bound components:")
    bound_info = kalai_vempala_bound(mfd, ece, n)
    print(f"  + MFd:                     {bound_info['term_mfd']:+.4f}")
    print(f"  - ECE:                     {bound_info['term_ece']:+.4f}")
    print(f"  - sparsity term:           {bound_info['term_sparsity']:+.6f}")
    print(f"  - sample term:             {bound_info['term_sample']:+.4f}")
    print(f"  -----------------------------------")
    print(f"  Lower bound on g(H):       {bound_info['bound']:.4f}")
    print()

    # Load observed hallucination rate
    with open("results/baseline_results.json") as f:
        baseline = json.load(f)
    observed = baseline["summary"]["hallucination_rate"]
    print(f"Observed hallucination rate: {observed:.4f}")
    print()

    # Interpret
    print("=" * 60)
    print("Interpretation")
    print("=" * 60)
    gap = bound_info["bound"] - observed
    print(f"Theoretical lower bound:     {bound_info['bound']:.4f}")
    print(f"Empirical rate:              {observed:.4f}")
    print(f"Gap:                         {gap:+.4f}")
    print()
    if gap > 0.1:
        print(f"The empirical rate is FAR BELOW the theoretical bound.")
        print(f"Per Kalai-Vempala, this means the model is NOT calibrated")
        print(f"in the sense required by the theorem. The bound only bites")
        print(f"when miscalibration is small — but our ECE = {ece:.3f} is")
        print(f"already large enough to invalidate it.")
        print()
        print(f"This is consistent with the 2025 follow-up paper: post-")
        print(f"trained models trade calibration for lower observed errors.")
    else:
        print(f"The empirical rate sits near the theoretical floor.")


if __name__ == "__main__":
    main()