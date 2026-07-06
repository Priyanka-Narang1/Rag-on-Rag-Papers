"""
Reciprocal Rank Fusion (RRF) hybrid retriever.
Combines BM25 and vector search results into a single ranked list.

RRF formula: score(d) = sum(1 / (k + rank(d))) for each retriever
where k=60 is a smoothing constant (standard value from the original
Cormack et al. 2009 paper -- not hardcoded arbitrarily, it is the
established default that works well empirically across retrieval tasks).

Why RRF over score normalization:
- BM25 and cosine distance scores are on completely different scales
  (BM25: 0-15+, cosine distance: 0-1) -- you cannot average them directly
- RRF only uses rank positions, so scale differences dont matter
- Simple, proven, no parameters to tune beyond k

Run: python src/hybrid_retriever.py
"""

import json
import pickle
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

DATA_DIR = Path(__file__).parent.parent / "data"
CHUNKS_FILE = DATA_DIR / "processed" / "chunks.jsonl"
BM25_INDEX_FILE = DATA_DIR / "processed" / "bm25_index.pkl"
CHROMA_DIR = DATA_DIR / "chroma_db"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "rag_papers"
RRF_K = 60  # standard RRF smoothing constant (Cormack et al. 2009)


def load_chunks():
    chunks = []
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return {"chunk_id": c["chunk_id"], "data": c} if False else {c["chunk_id"]: c for c in chunks}


def load_bm25():
    with open(BM25_INDEX_FILE, "rb") as f:
        payload = pickle.load(f)
    return payload["bm25"], payload["chunk_ids"]


def vector_search(collection, model, query, top_k=20):
    embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=embedding,
        n_results=top_k,
        include=["metadatas", "distances"]
    )
    ranked = []
    for chunk_id, dist in zip(results["ids"][0], results["distances"][0]):
        ranked.append({"chunk_id": chunk_id, "distance": dist})
    return ranked


def bm25_search(bm25, chunk_ids, query, top_k=20):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    ranked = []
    for idx in top_indices:
        ranked.append({"chunk_id": chunk_ids[idx], "bm25_score": scores[idx]})
    return ranked


def reciprocal_rank_fusion(vector_results, bm25_results, k=RRF_K):
    rrf_scores = {}

    for rank, result in enumerate(vector_results):
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank + 1)

    for rank, result in enumerate(bm25_results):
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank + 1)

    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_chunks


def hybrid_search(query, collection, model, bm25, chunk_ids, chunks_map, top_k=5):
    vector_results = vector_search(collection, model, query, top_k=20)
    bm25_results = bm25_search(bm25, chunk_ids, query, top_k=20)
    fused = reciprocal_rank_fusion(vector_results, bm25_results)

    final_results = []
    for chunk_id, rrf_score in fused[:top_k]:
        chunk = chunks_map.get(chunk_id, {})
        final_results.append({
            "chunk_id": chunk_id,
            "arxiv_id": chunk.get("arxiv_id", "unknown"),
            "rrf_score": rrf_score,
            "text": chunk.get("text", ""),
        })
    return final_results


def main():
    print("Loading resources...")
    chunks_map = load_chunks()
    bm25, chunk_ids = load_bm25()
    model = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    print("All resources loaded")

    test_queries = [
        "What is retrieval augmented generation?",
        "MRR@10 ColBERT passage retrieval",
        "REALM pre-training knowledge retriever",
        "How does self-RAG decide when to retrieve?",
    ]

    print("\n--- Hybrid retrieval sanity check (RRF k=60) ---")
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = hybrid_search(query, collection, model, bm25, chunk_ids, chunks_map, top_k=3)
        for i, r in enumerate(results):
            preview = r["text"][:100].replace("\n", " ")
            print(f"  {i+1}. [{r['arxiv_id']}] rrf={r['rrf_score']:.5f} | {preview}...")


if __name__ == "__main__":
    main()
