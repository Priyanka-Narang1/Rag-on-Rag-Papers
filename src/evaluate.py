import json
import os
import sys
import time
from pathlib import Path
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from groq import Groq

DATA_DIR = Path(__file__).parent.parent / "data"
EVAL_DIR = Path(__file__).parent.parent / "eval"
CHUNKS_FILE = DATA_DIR / "processed" / "chunks.jsonl"
BM25_INDEX_FILE = DATA_DIR / "processed" / "bm25_index.pkl"
CHROMA_DIR = DATA_DIR / "chroma_db"
GOLDEN_DATASET = EVAL_DIR / "golden_dataset.json"
RESULTS_FILE = EVAL_DIR / "eval_results.json"

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
    results = collection.query(query_embeddings=embedding, n_results=top_k, include=["distances"])
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
    vr = vector_search(collection, embed_model, query, top_k=RERANK_POOL)
    br = bm25_search(bm25, chunk_ids, query, top_k=RERANK_POOL)
    fused = reciprocal_rank_fusion(vr, br)
    candidate_ids = [cid for cid, _ in fused[:RERANK_POOL]]
    return rerank(query, candidate_ids, chunks_map, reranker, top_k=FINAL_TOP_K)


def groq_json(system, user, client):
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        response_format={"type": "json_object"},
        temperature=0.0,
        max_tokens=512,
    )
    return response.choices[0].message.content


def generate_answer(question, chunks, client):
    context = "\n\n---\n\n".join(f"[{c['chunk_id']}]\n{c['text']}" for c in chunks)
    system = (
        "Answer using ONLY the context. Output JSON only with keys answer and citations. "
        'Example: {"answer": "your answer", "citations": ["chunk_id"]}'
    )
    raw = groq_json(system, f"Context:\n{context}\n\nQuestion: {question}", client)
    try:
        return json.loads(raw).get("answer", raw)
    except Exception:
        return raw


def score_faithfulness(question, answer, contexts, client):
    context_text = "\n\n".join(contexts)
    system_claims = (
        "Extract all factual claims from the answer below. "
        "Return JSON with key claims containing a list of strings. "
        'Example output: {"claims": ["claim one", "claim two"]}'
    )
    claims_raw = groq_json(system_claims, f"Answer: {answer}", client)
    try:
        claims = json.loads(claims_raw).get("claims", [])
    except Exception:
        return 0.0
    if not claims:
        return 1.0
    supported = 0
    system_verify = (
        "You are a fact checker. Given context and a claim, return JSON with key supported (boolean). "
        'Example: {"supported": true} or {"supported": false}'
    )
    for claim in claims:
        verdict_raw = groq_json(
            system_verify,
            f"Context:\n{context_text}\n\nClaim: {claim}",
            client
        )
        try:
            if json.loads(verdict_raw).get("supported", False):
                supported += 1
        except Exception:
            pass
        time.sleep(0.3)
    return round(supported / len(claims), 4)


def score_answer_relevancy(question, answer, embed_model):
    if "cannot answer" in answer.lower():
        return 0.0
    q_emb = embed_model.encode([question])
    a_emb = embed_model.encode([answer])
    sim = float(
        np.dot(q_emb[0], a_emb[0]) /
        (np.linalg.norm(q_emb[0]) * np.linalg.norm(a_emb[0]) + 1e-9)
    )
    return round(max(0.0, sim), 4)


def score_context_precision(question, contexts, client):
    if not contexts:
        return 0.0
    system_rel = (
        "You are a relevance judge. Given a question and a context passage, "
        "return JSON with key relevant (boolean) indicating if the context "
        'helps answer the question. Example: {"relevant": true}'
    )
    relevant = 0
    for ctx in contexts:
        verdict_raw = groq_json(
            system_rel,
            f"Question: {question}\n\nContext: {ctx[:500]}",
            client
        )
        try:
            if json.loads(verdict_raw).get("relevant", False):
                relevant += 1
        except Exception:
            pass
        time.sleep(0.3)
    return round(relevant / len(contexts), 4)


def main():
    run_all = "--all" in sys.argv
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set.")
    client = Groq(api_key=api_key)

    print("Loading resources...")
    chunks_map = load_chunks()
    bm25, chunk_ids = load_bm25()
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_collection(COLLECTION_NAME)
    print("All resources loaded\n")

    with open(GOLDEN_DATASET, encoding="utf-8-sig") as f:
        golden = json.load(f)

    questions_to_run = golden if run_all else golden[:10]
    print(f"Evaluating {len(questions_to_run)} questions...\n")

    all_faithfulness = []
    all_relevancy = []
    all_precision = []
    per_question_results = []

    for i, item in enumerate(questions_to_run):
        q = item["question"]
        print(f"[{i+1}/{len(questions_to_run)}] {q[:65]}...")
        top_chunks = retrieve(q, collection, embed_model, reranker, bm25, chunk_ids, chunks_map)
        contexts = [c["text"] for c in top_chunks]
        answer = generate_answer(q, top_chunks, client)
        time.sleep(0.5)
        faith = score_faithfulness(q, answer, contexts, client)
        rel = score_answer_relevancy(q, answer, embed_model)
        prec = score_context_precision(q, contexts, client)
        all_faithfulness.append(faith)
        all_relevancy.append(rel)
        all_precision.append(prec)
        per_question_results.append({
            "id": item["id"],
            "question": q,
            "answer": answer,
            "faithfulness": faith,
            "answer_relevancy": rel,
            "context_precision": prec,
        })
        print(f"   faith={faith} | relevancy={rel} | precision={prec}")
        time.sleep(0.5)

    print("\n" + "="*55)
    print("EVALUATION RESULTS")
    print("="*55)
    final_scores = {
        "faithfulness": round(float(np.mean(all_faithfulness)), 4),
        "answer_relevancy": round(float(np.mean(all_relevancy)), 4),
        "context_precision": round(float(np.mean(all_precision)), 4),
        "num_questions": len(questions_to_run),
    }
    for k, v in final_scores.items():
        print(f"  {k}: {v}")

    output = {"summary": final_scores, "per_question": per_question_results}
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {RESULTS_FILE}")
    print("These are your CI baseline scores.")


if __name__ == "__main__":
    main()
