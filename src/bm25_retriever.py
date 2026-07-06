"""
BM25 retriever over the 602 chunks.
Builds a BM25 index from chunks.jsonl and supports keyword search.

BM25 complements vector search:
- Vector search: good at semantic similarity ("what is dense retrieval")
- BM25: good at exact keyword matches ("MRR@10", "REALM", specific author names)

Run: python src/bm25_retriever.py
"""

import json
import pickle
from pathlib import Path
from rank_bm25 import BM25Okapi

DATA_DIR = Path(__file__).parent.parent / "data"
CHUNKS_FILE = DATA_DIR / "processed" / "chunks.jsonl"
BM25_INDEX_FILE = DATA_DIR / "processed" / "bm25_index.pkl"


def load_chunks():
    chunks = []
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def tokenize(text):
    # Simple whitespace + lowercase tokenization
    # Good enough for BM25 -- we are not doing stemming or stopword removal
    # deliberately so exact technical terms (e.g. "MRR@10", "ColBERT") match precisely
    return text.lower().split()


def build_bm25_index(chunks):
    print(f"Building BM25 index over {len(chunks)} chunks...")
    corpus = [tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus)
    return bm25


def save_index(bm25, chunks):
    payload = {"bm25": bm25, "chunk_ids": [c["chunk_id"] for c in chunks]}
    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump(payload, f)
    print(f"BM25 index saved to {BM25_INDEX_FILE}")


def search_bm25(bm25, chunks, query, top_k=5):
    tokens = tokenize(query)
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for idx in top_indices:
        results.append({
            "chunk_id": chunks[idx]["chunk_id"],
            "arxiv_id": chunks[idx]["arxiv_id"],
            "bm25_score": scores[idx],
            "text": chunks[idx]["text"],
        })
    return results


def main():
    chunks = load_chunks()
    bm25 = build_bm25_index(chunks)
    save_index(bm25, chunks)

    # Sanity check with 3 different query types
    test_queries = [
        "What is retrieval augmented generation?",   # semantic/broad
        "MRR@10 ColBERT passage retrieval",          # exact technical terms
        "REALM pre-training knowledge retriever",    # specific paper concepts
    ]

    print("\n--- BM25 sanity check ---")
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = search_bm25(bm25, chunks, query, top_k=3)
        for i, r in enumerate(results):
            preview = r["text"][:100].replace("\n", " ")
            print(f"  {i+1}. [{r['arxiv_id']}] score={r['bm25_score']:.4f} | {preview}...")

    print("\nBM25 index ready.")


if __name__ == "__main__":
    main()
