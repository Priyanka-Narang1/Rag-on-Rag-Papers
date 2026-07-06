"""
Answer generation with citation enforcement using Groq.
Prompt loaded from prompts/v1.yaml -- version tracked per response.
"""

import json
import os
from pathlib import Path
import pickle
import yaml
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from groq import Groq

DATA_DIR = Path(__file__).parent.parent / "data"
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"
CHUNKS_FILE = DATA_DIR / "processed" / "chunks.jsonl"
BM25_INDEX_FILE = DATA_DIR / "processed" / "bm25_index.pkl"
CHROMA_DIR = DATA_DIR / "chroma_db"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "rag_papers"
PROMPT_VERSION = "v1"
RRF_K = 60
RERANK_POOL = 30
FINAL_TOP_K = 5


def load_prompt_config(version=PROMPT_VERSION):
    config_path = PROMPTS_DIR / f"{version}.yaml"
    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


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


def build_prompt(query, chunks, config):
    context_parts = []
    for chunk in chunks:
        context_parts.append(f"[{chunk['chunk_id']}]\n{chunk['text']}")
    context = "\n\n---\n\n".join(context_parts)
    system = config["system_prompt"]
    user = f"Context:\n{context}\n\nQuestion: {query}"
    return system, user


def call_groq(system, user, client, config):
    response = client.chat.completions.create(
        model=config["model"],
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        response_format={"type": "json_object"},
        temperature=config["temperature"],
        max_tokens=config["max_tokens"],
    )
    return response.choices[0].message.content, response.usage


def parse_and_verify_citations(raw_response, retrieved_chunk_ids):
    try:
        parsed = json.loads(raw_response)
    except json.JSONDecodeError:
        return {
            "answer": raw_response,
            "citations": [],
            "hallucinated_citations": [],
            "parse_error": True,
        }
    answer = parsed.get("answer", "")
    cited = parsed.get("citations", [])
    retrieved_set = set(retrieved_chunk_ids)
    valid_citations = [c for c in cited if c in retrieved_set]
    hallucinated = [c for c in cited if c not in retrieved_set]
    return {
        "answer": answer,
        "citations": valid_citations,
        "hallucinated_citations": hallucinated,
        "parse_error": False,
    }


def ask(query, collection, embed_model, reranker, bm25, chunk_ids, chunks_map, client, config):
    top_chunks = retrieve(query, collection, embed_model, reranker, bm25, chunk_ids, chunks_map)
    retrieved_ids = [c["chunk_id"] for c in top_chunks]
    system, user = build_prompt(query, top_chunks, config)
    raw_response, usage = call_groq(system, user, client, config)
    result = parse_and_verify_citations(raw_response, retrieved_ids)
    result["query"] = query
    result["retrieved_chunks"] = retrieved_ids
    result["prompt_version"] = config["version"]
    result["tokens_used"] = {
        "prompt": usage.prompt_tokens,
        "completion": usage.completion_tokens,
        "total": usage.total_tokens,
    }
    return result


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set.")

    config = load_prompt_config()
    print(f"Loaded prompt version: {config['version']} -- {config['description']}")

    client = Groq(api_key=api_key)

    print("Loading resources...")
    chunks_map = load_chunks()
    bm25, chunk_ids = load_bm25()
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_collection(COLLECTION_NAME)
    print("All resources loaded\n")

    test_queries = [
        "What evaluation metrics does RAGAS use?",
        "How does RAPTOR organize documents differently from flat chunking?",
        "How does self-RAG decide when to retrieve?",
    ]

    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        result = ask(query, collection, embed_model, reranker, bm25, chunk_ids, chunks_map, client, config)
        print(f"Prompt version: {result['prompt_version']}")
        print(f"\nAnswer:\n{result['answer']}")
        print(f"\nCitations: {result['citations']}")
        if result["hallucinated_citations"]:
            print(f"HALLUCINATED CITATIONS: {result['hallucinated_citations']}")
        else:
            print("Citation check: PASSED")
        print(f"Tokens: {result['tokens_used']}")


if __name__ == "__main__":
    main()
