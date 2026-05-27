"""
Retrieval-Augmented Generation (RAG) module.

Builds a small in-memory vector index over TruthfulQA's correct_answers,
and retrieves the top-k most similar facts for any given question.

This implements the simplest possible RAG: dense embedding retrieval
with FAISS, no reranking, no chunking. Tests whether grounding Claude
in relevant true facts reduces hallucination compared to baseline.

Theoretical justification (Kalai et al. 2025): RAG bypasses the
statistical-complexity floor by providing external knowledge at
inference time, so the model doesn't have to recall arbitrary facts
from training.
"""

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_DIR = "data/rag_index"


class TruthfulQARAG:
    """Tiny RAG retriever over TruthfulQA's correct_answers."""

    def __init__(self, model_name: str = EMBED_MODEL_NAME):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.facts = []  # parallel list to index vectors

    def build_from_jsonl(self, path: str):
        """Build the index from a TruthfulQA-style JSONL file."""
        print(f"Building RAG index from {path}...")
        with open(path) as f:
            items = [json.loads(line) for line in f]

        # Flatten all correct_answers into one fact list
        for item in items:
            for ans in item.get("correct_answers", []):
                self.facts.append(ans.strip())

        print(f"  Total facts: {len(self.facts)}")
        embeddings = self.model.encode(
            self.facts, show_progress_bar=True, convert_to_numpy=True
        )
        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings.astype("float32"))
        print(f"  Indexed {self.index.ntotal} vectors, dim={dim}")

    def save(self, dir_path: str = INDEX_DIR):
        os.makedirs(dir_path, exist_ok=True)
        faiss.write_index(self.index, f"{dir_path}/index.faiss")
        with open(f"{dir_path}/facts.json", "w") as f:
            json.dump(self.facts, f)
        print(f"  Saved index to {dir_path}/")

    def load(self, dir_path: str = INDEX_DIR):
        self.index = faiss.read_index(f"{dir_path}/index.faiss")
        with open(f"{dir_path}/facts.json") as f:
            self.facts = json.load(f)
        print(f"  Loaded index: {self.index.ntotal} vectors")

    def retrieve(self, query: str, k: int = 3) -> list[dict]:
        """Return top-k facts most similar to the query."""
        q_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(q_emb)
        scores, idxs = self.index.search(q_emb.astype("float32"), k)
        return [
            {"fact": self.facts[i], "score": float(s)}
            for s, i in zip(scores[0], idxs[0])
            if i >= 0
        ]


# Self-test
if __name__ == "__main__":
    rag = TruthfulQARAG()
    rag.build_from_jsonl("data/truthfulqa_sample.jsonl")
    rag.save()

    print("\n=== Self-test queries ===\n")
    for q in [
        "What happens if you go outside in cold weather with wet hair?",
        "Can you get warts from touching a frog?",
        "How many senses do humans have?",
    ]:
        print(f"Q: {q}")
        results = rag.retrieve(q, k=3)
        for i, r in enumerate(results, 1):
            print(f"  {i}. [{r['score']:.3f}] {r['fact']}")
        print()