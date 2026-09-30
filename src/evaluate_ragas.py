"""
Ragas evaluation script.
Runs a subset of the golden dataset through the full RAG pipeline
and scores using Ragas metrics:
  - faithfulness: is the answer supported by the retrieved context?
  - answer_relevancy: does the answer address the question?
  - context_precision: are the retrieved chunks actually relevant?

We run 10 questions (not all 60) for the first eval run to keep
API costs low and confirm the pipeline works. Once confirmed,
run with --all flag for the full 60.

Run: python src/evaluate_ragas.py
Run all: python src/evaluate_ragas.py --all
"""

import json
import os
import sys
import time
from pathlib import Path
import pickle
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from groq import Groq
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

DATA_DIR = Path(__file__).parent.parent / "data"
EVAL_DIR = Path(__file__).parent.parent / "eval"
CHUNKS_FILE = DATA_DIR / "processed" / "chunks.jsonl"
BM25_INDEX_FILE = DATA_DIR / "processed" / "bm25_index.pkl"
CHROMA_DIR = DATA_DIR / "chroma_db"
GOLDEN_DATASET = EVAL_DIR / "golden_dataset.json"
RESULTS_FILE = EVAL_DIR / "ragas_results.json"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "rag_papers"
GROQ_MODEL = "llama3-8b-8192"
RRF_K = 60
RERANK_POOL = 30
FINAL_TOP_K = 5


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


def vector_search(collection, model, query, top_k=30):
    embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=embedding,
        n_results=top_k,
        include=["metadatas", "distances"]
    )
    return [{"chunk_id": cid} for cid in results["ids"][0]]


def bm25_search(bm25, chunk_ids, query, top_k=30):
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [{"chunk_id": chunk_ids[idx]} for idx in top_indices]


def reciprocal_rank_fusion(vector_results, bm25_results, k=RRF_K):
    rrf_scores = {}
    for rank, r in enumerate(vector_results):
        cid = r["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank + 1)
    for rank, r in enumerate(bm25_results):
        cid = r["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (k + rank + 1)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)


def rerank(query, candidate_ids, chunks_map, reranker, top_k=FINAL_TOP_K):
    pairs = [(query, chunks_map[cid]["text"]) for cid in candidate_ids if cid in chunks_map]
    scores = reranker.predict(pairs)
    scored = sorted(zip(candidate_ids, scores), key=lambda x: x[1], reverse=True)
    return [chunks_map[cid] for cid, _ in scored[:top_k] if cid in chunks_map]


def retrieve(query, collection, embed_model, reranker, bm25, chunk_ids, chunks_map):
    vector_results = vector_search(collection, embed_model, query, top_k=RERANK_POOL)
    bm25_results = bm25_search(bm25, chunk_ids, query, top_k=RERANK_POOL)
    fused = reciprocal_rank_fusion(vector_results, bm25_results)
    candidate_ids = [cid for cid, _ in fused[:RERANK_POOL]]
    return rerank(query, candidate_ids, chunks_map, reranker, top_k=FINAL_TOP_K)


def generate_answer(query, chunks, groq_client):
    context_parts = []
    for chunk in chunks:
        context_parts.append(f"[{chunk['chunk_id']}]\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)

    system = """You are a research assistant answering questions about AI/ML papers.
Answer ONLY using the provided context chunks.
Output ONLY valid JSON: {"answer": "your answer", "citations": ["chunk_id_1"]}
If context is insufficient: {"answer": "I cannot answer this from the provided context.", "citations": []}"""

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=512,
    )
    raw = response.choices[0].message.content
    try:
        parsed = json.loads(raw)
        return parsed.get("answer", raw)
    except json.JSONDecodeError:
        return raw


def main():
    run_all = "--all" in sys.argv

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set.")

    groq_client = Groq(api_key=api_key)

    print("Loading resources...")
    chunks_map = load_chunks()
    bm25, chunk_ids = load_bm25()
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_collection(COLLECTION_NAME)
    print("All resources loaded\n")

    with open(GOLDEN_DATASET, encoding="utf-8") as f:
        golden = json.load(f)

    questions_to_run = golden if run_all else golden[:10]
    print(f"Running Ragas eval on {len(questions_to_run)} questions...")

    eval_rows = []
    for i, item in enumerate(questions_to_run):
        print(f"  [{i+1}/{len(questions_to_run)}] {item['question'][:60]}...")
        top_chunks = retrieve(item["question"], collection, embed_model, reranker, bm25, chunk_ids, chunks_map)
        contexts = [c["text"] for c in top_chunks]
        answer = generate_answer(item["question"], top_chunks, groq_client)

        eval_rows.append({
            "question": item["question"],
            "answer": answer,
            "contexts": contexts,
            "ground_truth": item["ground_truth"],
        })
        time.sleep(0.5)  # avoid Groq rate limits

    dataset = Dataset.from_list(eval_rows)

    print("\nRunning Ragas scoring...")
    results = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision],
    )

    print("\n" + "="*50)
    print("RAGAS EVALUATION RESULTS")
    print("="*50)
    scores = {
        "faithfulness": round(float(results["faithfulness"]), 4),
        "answer_relevancy": round(float(results["answer_relevancy"]), 4),
        "context_precision": round(float(results["context_precision"]), 4),
        "num_questions": len(questions_to_run),
    }
    for metric, score in scores.items():
        print(f"  {metric}: {score}")

    with open(RESULTS_FILE, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"\nResults saved to {RESULTS_FILE}")
    print("These scores become your CI regression baseline.")


if __name__ == "__main__":
    main()
