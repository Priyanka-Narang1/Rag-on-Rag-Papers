"""
Cross-encoder reranker using ms-marco-MiniLM-L-6-v2.

Why cross-encoders improve on RRF:
- RRF fusion uses rank positions only -- it has no understanding of
  whether a chunk actually ANSWERS the query
- Cross-encoders take (query, chunk) as a pair and score relevance
  directly -- much higher precision but slower (runs on top-k only)
- We run cross-encoder only on top 20 RRF results, then rerank to top 5
  This keeps latency acceptable while maximizing precision

Model: cross-encoder/ms-marco-MiniLM-L-6-v2
- Trained on MS MARCO passage ranking (180k+ queries)
- Fast (MiniLM architecture), good quality for passage reranking
- Score range: roughly -10 to +10 (higher = more relevant)

Run: python src/reranker.py
"""

from pathlib import Path
from sentence_transformers import CrossEncoder
import json
import pickle
from sentence_transformers import SentenceTransformer
import chromadb

DATA_DIR = Path(__file__).parent.parent / "data"
CHUNKS_FILE = DATA_DIR / "processed" / "chunks.jsonl"
BM25_INDEX_FILE = DATA_DIR / "processed" / "bm25_index.pkl"
CHROMA_DIR = DATA_DIR / "chroma_db"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "rag_papers"
RRF_K = 60
RERANK_POOL = 20   # how many RRF results to feed into reranker
FINAL_TOP_K = 5    # how many to return after reranking


def load_chunks():
    chunks = {}
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunks[c["chunk_id"]] = c
    return chunks


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
    return [{"chunk_id": cid, "distance": dist}
            for cid, dist in zip(results["ids"][0], results["distances"][0])]


def bm25_search(bm25, chunk_ids, query, top_k=20):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [{"chunk_id": chunk_ids[idx], "bm25_score": scores[idx]}
            for idx in top_indices]


def reciprocal_rank_fusion(vector_results, bm25_results, k=RRF_K):
    rrf_scores = {}
    for rank, r in enumerate(vector_results):
        cid = r["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank + 1)
    for rank, r in enumerate(bm25_results):
        cid = r["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank + 1)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def rerank(query, candidate_chunk_ids, chunks_map, reranker, top_k=FINAL_TOP_K):
    pairs = [(query, chunks_map[cid]["text"]) for cid in candidate_chunk_ids if cid in chunks_map]
    scores = reranker.predict(pairs)
    scored = sorted(zip(candidate_chunk_ids, scores), key=lambda x: x[1], reverse=True)
    results = []
    for cid, score in scored[:top_k]:
        chunk = chunks_map[cid]
        results.append({
            "chunk_id": cid,
            "arxiv_id": chunk["arxiv_id"],
            "rerank_score": float(score),
            "text": chunk["text"],
        })
    return results


def full_pipeline(query, collection, embed_model, reranker, bm25, chunk_ids, chunks_map):
    # Stage 1: hybrid retrieval (top 20 candidates)
    vector_results = vector_search(collection, embed_model, query, top_k=RERANK_POOL)
    bm25_results = bm25_search(bm25, chunk_ids, query, top_k=RERANK_POOL)
    fused = reciprocal_rank_fusion(vector_results, bm25_results)
    candidate_ids = [cid for cid, _ in fused[:RERANK_POOL]]

    # Stage 2: cross-encoder reranking (top 5 from candidates)
    final_results = rerank(query, candidate_ids, chunks_map, reranker, top_k=FINAL_TOP_K)
    return final_results


def main():
    print("Loading resources...")
    chunks_map = load_chunks()
    bm25, chunk_ids = load_bm25()
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    print("All resources loaded\n")

    test_queries = [
        "What is retrieval augmented generation?",
        "How does self-RAG decide when to retrieve?",
        "What chunking strategy does RAPTOR use?",
        "How does ColBERT differ from dense passage retrieval?",
    ]

    print("--- Full pipeline: hybrid retrieval + cross-encoder reranking ---")
    for query in test_queries:
        print(f"\nQuery: {query}")
        results = full_pipeline(query, collection, embed_model, reranker, bm25, chunk_ids, chunks_map)
        for i, r in enumerate(results):
            preview = r["text"][:100].replace("\n", " ")
            print(f"  {i+1}. [{r['arxiv_id']}] rerank_score={r['rerank_score']:.4f} | {preview}...")


if __name__ == "__main__":
    main()
