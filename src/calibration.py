"""
Expected Calibration Error (ECE) computation.

Following the standard definition (Naeini et al. 2015; Guo et al. 2017),
and the framing used in Kalai-Vempala (2024) Figure 1 and Kalai et al. (2025).

ECE measures the gap between predicted confidence and actual accuracy,
binned by confidence level. ECE = 0 means perfect calibration.
"""

import json
import numpy as np


def expected_calibration_error(
    confidences: list[float],
    correctness: list[int],
    n_bins: int = 10,
) -> dict:
    """
    Compute ECE given lists of confidence scores and correctness labels.

    Args:
        confidences: list of floats in [0, 1]
        correctness: list of 0/1 (1 = correct/truthful, 0 = hallucinated)
        n_bins: number of confidence bins (default 10, i.e., 0.0-0.1, 0.1-0.2, ...)

    Returns:
        dict with ECE, per-bin stats, and metadata
    """
    confidences = np.array(confidences, dtype=float)
    correctness = np.array(correctness, dtype=float)
    n = len(confidences)

    if n == 0:
        return {"ece": None, "bins": [], "n": 0}

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bins = []

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        # Include left edge; include right edge on last bin only
        if i == n_bins - 1:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)

        bin_size = mask.sum()
        if bin_size > 0:
            bin_conf = confidences[mask].mean()
            bin_acc = correctness[mask].mean()
            gap = abs(bin_conf - bin_acc)
            ece += (bin_size / n) * gap
            bins.append({
                "range": [round(lo, 2), round(hi, 2)],
                "count": int(bin_size),
                "avg_confidence": round(float(bin_conf), 4),
                "accuracy": round(float(bin_acc), 4),
                "gap": round(float(gap), 4),
            })
        else:
            bins.append({
                "range": [round(lo, 2), round(hi, 2)],
                "count": 0,
                "avg_confidence": None,
                "accuracy": None,
                "gap": None,
            })

    return {
        "ece": round(float(ece), 4),
        "n": n,
        "n_bins": n_bins,
        "bins": bins,
    }


def compute_from_results_file(path: str) -> dict:
    """
    Load a results JSON file and compute ECE.
    Expects results with 'confidence' and 'truthful' fields per item.
    """
    with open(path) as f:
        data = json.load(f)

    results = data.get("results", [])
    valid = [
        r for r in results
        if r.get("confidence") is not None and r.get("truthful") is not None
    ]

    if not valid:
        return {
            "ece": None,
            "n": 0,
            "note": "No items with both 'confidence' and 'truthful' fields.",
        }

    confidences = [r["confidence"] for r in valid]
    correctness = [r["truthful"] for r in valid]
    return expected_calibration_error(confidences, correctness)


if __name__ == "__main__":
    # Self-test with a synthetic example
    print("=== Self-test: perfect calibration ===")
    confs = [0.1] * 10 + [0.9] * 10
    correct = [0] * 9 + [1] + [1] * 9 + [0]  # 10% and 90% accuracy
    result = expected_calibration_error(confs, correct, n_bins=10)
    print(f"ECE: {result['ece']} (should be near 0)")
    print()

    print("=== Self-test: overconfident model ===")
    confs = [0.9] * 20
    correct = [1] * 5 + [0] * 15  # says 90% but only 25% right
    result = expected_calibration_error(confs, correct, n_bins=10)
    print(f"ECE: {result['ece']} (should be ~0.65)")