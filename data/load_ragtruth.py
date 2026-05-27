"""
Load the RAGTruth dataset directly from the official GitHub repository
(ParticleMedia/RAGTruth). RAGTruth (Niu et al., ACL 2024) contains nearly
18,000 hallucination-annotated LLM responses from RAG tasks.

Per Kalai et al. 2025: this dataset is theoretically a better testbed
for data normalization than TruthfulQA, because retrieved evidence chunks
contain duplicate facts -- giving normalization something real to collapse.
"""

import json
import os
import urllib.request
from collections import Counter

# Official RAGTruth raw URLs
RAGTRUTH_RESPONSE_URL = (
    "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/"
    "main/dataset/response.jsonl"
)
RAGTRUTH_SOURCE_URL = (
    "https://raw.githubusercontent.com/ParticleMedia/RAGTruth/"
    "main/dataset/source_info.jsonl"
)

LOCAL_RESPONSE_PATH = "data/ragtruth_response.jsonl"
LOCAL_SOURCE_PATH = "data/ragtruth_source.jsonl"
SAMPLE_PATH = "data/ragtruth_sample.jsonl"
SAMPLE_SIZE = 50


def download(url: str, dest: str):
    """Download a file with a basic progress message."""
    print(f"Downloading {url}")
    print(f"  -> {dest}")
    urllib.request.urlretrieve(url, dest)
    size_mb = os.path.getsize(dest) / 1024 / 1024
    print(f"  Done ({size_mb:.1f} MB)")


def load_jsonl(path: str) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def main():
    os.makedirs("data", exist_ok=True)

    # Download if not already present
    if not os.path.exists(LOCAL_RESPONSE_PATH):
        download(RAGTRUTH_RESPONSE_URL, LOCAL_RESPONSE_PATH)
    if not os.path.exists(LOCAL_SOURCE_PATH):
        download(RAGTRUTH_SOURCE_URL, LOCAL_SOURCE_PATH)

    responses = load_jsonl(LOCAL_RESPONSE_PATH)
    sources = load_jsonl(LOCAL_SOURCE_PATH)
    print(f"\nLoaded {len(responses)} responses and {len(sources)} source records.\n")

    # Show structure of first record
    print("=== Sample response record ===")
    sample = responses[0]
    for k, v in sample.items():
        v_str = str(v)
        if len(v_str) > 200:
            v_str = v_str[:200] + "..."
        print(f"  {k}: {v_str}")
    print()

    print("=== Sample source record ===")
    sample_src = sources[0]
    for k, v in sample_src.items():
        v_str = str(v)
        if len(v_str) > 200:
            v_str = v_str[:200] + "..."
        print(f"  {k}: {v_str}")
    print()

    # Save sample
    with open(SAMPLE_PATH, "w") as f:
        for r in responses[:SAMPLE_SIZE]:
            f.write(json.dumps(r) + "\n")
    print(f"Saved {SAMPLE_SIZE} responses to {SAMPLE_PATH}\n")

    # MonoFact analysis on response text
    print("=== MonoFact analysis on response text (first 500 entries) ===")
    text_field = None
    for candidate in ["response", "output", "answer", "model_response"]:
        if candidate in responses[0]:
            text_field = candidate
            break

    if text_field:
        sample_for_mf = [str(r.get(text_field, "")) for r in responses[:500]]
        counts = Counter(sample_for_mf)
        n = len(sample_for_mf)
        unique = len(counts)
        monofacts = sum(1 for c in counts.values() if c == 1)
        print(f"  Field used:       {text_field}")
        print(f"  Total responses:  {n}")
        print(f"  Unique:           {unique}")
        print(f"  Monofacts:        {monofacts}")
        print(f"  Monofact rate:    {monofacts / n:.4f}")
        print()
        print(f"  Compare TruthfulQA: 0.9895")
        print(f"  Compare RAGTruth:   {monofacts / n:.4f}")
        if monofacts / n < 0.9895:
            print(
                f"\n  -> RAGTruth has a LOWER monofact rate -- the theory "
                f"predicts normalization should help here more than on "
                f"TruthfulQA. Phase 2 target validated."
            )
        else:
            print(
                f"\n  -> RAGTruth monofact rate is similar to TruthfulQA. "
                f"Need to check different fields or aggregation."
            )
    else:
        print("  No standard response field found. Available fields:")
        print(f"  {list(responses[0].keys())}")


if __name__ == "__main__":
    main()