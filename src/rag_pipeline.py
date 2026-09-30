import json
import os
import time
from pathlib import Path
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from groq import Groq
from langfuse import Langfuse
import yaml

DATA_DIR = Path(__file__).parent.parent / "data"
CHUNKS_FILE = DATA_DIR / "processed" / "chunks.jsonl"
BM25_INDEX_FILE = DATA_DIR / "processed" / "bm25_index.pkl"
CHROMA_DIR = DATA_DIR / "chroma_db"
PROMPTS_DIR = Path(__file__).parent.parent / "prompts"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "rag_papers"
GROQ_MODEL = "llama3-8b-8192"
RRF_K = 60
RERANK_POOL = 30
FINAL_TOP_K = 5
COST_PER_1K_INPUT = 0.00005
COST_PER_1K_OUTPUT = 0.00008


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


def load_prompt_config():
    with open(PROMPTS_DIR / "v1.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def init_langfuse():
    try:
        lf = Langfuse(
            secret_key=os.environ.get("LANGFUSE_SECRET_KEY"),
            public_key=os.environ.get("LANGFUSE_PUBLIC_KEY"),
            host="https://cloud.langfuse.com"
        )
        if lf.auth_check():
            print("[Langfuse] connected")
            return lf
        return None
    except Exception as e:
        print(f"[Langfuse] init failed: {e}")
        return None


def rag_query(query, collection, embed_model, reranker, bm25, chunk_ids,
              chunks_map, groq_client, prompt_config, lf=None):

    latencies = {}
    total_start = time.perf_counter()

    with (lf.start_as_current_observation(name="rag_query", input={"query": query})
          if lf else __import__("contextlib").nullcontext()) as root_obs:

        # Stage 1: embed
        t0 = time.perf_counter()
        query_embedding = embed_model.encode([query]).tolist()
        latencies["embed_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Stage 2: vector search
        t0 = time.perf_counter()
        vr = collection.query(query_embeddings=query_embedding,
                              n_results=RERANK_POOL, include=["distances"])
        vector_results = [{"chunk_id": cid} for cid in vr["ids"][0]]
        latencies["vector_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Stage 3: BM25
        t0 = time.perf_counter()
        tokens = query.lower().split()
        scores = bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:RERANK_POOL]
        bm25_results = [{"chunk_id": chunk_ids[idx]} for idx in top_indices]
        latencies["bm25_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        # Stage 4: RRF fusion
        rrf_scores = {}
        for rank, r in enumerate(vector_results):
            cid = r["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
        for rank, r in enumerate(bm25_results):
            cid = r["chunk_id"]
            rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (RRF_K + rank + 1)
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        candidate_ids = [cid for cid, _ in fused[:RERANK_POOL]]

        # Stage 5: rerank
        t0 = time.perf_counter()
        pairs = [(query, chunks_map[cid]["text"]) for cid in candidate_ids if cid in chunks_map]
        rerank_scores = reranker.predict(pairs)
        scored = sorted(zip(candidate_ids, rerank_scores), key=lambda x: x[1], reverse=True)
        top_chunks = [chunks_map[cid] for cid, _ in scored[:FINAL_TOP_K] if cid in chunks_map]
        latencies["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        retrieved_ids = [c["chunk_id"] for c in top_chunks]

        # Log retrieval span
        if lf:
            try:
                with lf.start_as_current_observation(
                    name="retrieval",
                    input={"query": query},
                ) as span:
                    span.update(
                        output={"chunk_ids": retrieved_ids},
                        metadata={"latencies_ms": latencies}
                    )
            except Exception:
                pass

        # Stage 6: LLM
        context = "\n\n---\n\n".join(
            f"[{c['chunk_id']}]\n{c['text']}" for c in top_chunks
        )
        system = prompt_config.get("system_prompt", "")
        user = f"Context:\n{context}\n\nQuestion: {query}"

        t0 = time.perf_counter()
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            response_format={"type": "json_object"},
            temperature=prompt_config.get("temperature", 0.1),
            max_tokens=prompt_config.get("max_tokens", 1024),
        )
        latencies["llm_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        raw = response.choices[0].message.content
        usage = response.usage
        prompt_tokens = usage.prompt_tokens
        completion_tokens = usage.completion_tokens
        cost_usd = round(
            (prompt_tokens / 1000 * COST_PER_1K_INPUT) +
            (completion_tokens / 1000 * COST_PER_1K_OUTPUT), 6
        )

        try:
            parsed = json.loads(raw)
            answer = parsed.get("answer", raw)
            cited = parsed.get("citations", [])
        except Exception:
            answer = raw
            cited = []

        retrieved_set = set(retrieved_ids)
        valid_citations = [c for c in cited if c in retrieved_set]
        hallucinated = [c for c in cited if c not in retrieved_set]
        latencies["total_ms"] = round((time.perf_counter() - total_start) * 1000, 2)

        # Log LLM generation span
        if lf:
            try:
                with lf.start_as_current_observation(name="llm_generation") as gen:
                    gen.update(
                        input={"system": system[:200], "user": user[:200]},
                        output=answer,
                        metadata={
                            "model": GROQ_MODEL,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": completion_tokens,
                            "cost_usd": cost_usd,
                            "latencies_ms": latencies,
                            "valid_citations": valid_citations,
                            "hallucinated_citations": hallucinated,
                        }
                    )
            except Exception:
                pass

        if root_obs and lf:
            try:
                root_obs.update(
                    output={"answer": answer, "citations": valid_citations},
                    metadata={"latencies_ms": latencies, "cost_usd": cost_usd}
                )
            except Exception:
                pass

    return {
        "query": query,
        "answer": answer,
        "citations": valid_citations,
        "hallucinated_citations": hallucinated,
        "retrieved_chunks": retrieved_ids,
        "latencies_ms": latencies,
        "tokens": {"prompt": prompt_tokens, "completion": completion_tokens},
        "cost_usd": cost_usd,
        "prompt_version": prompt_config.get("version"),
    }


def compute_percentiles(values, label):
    arr = np.array(values)
    p50 = np.percentile(arr, 50)
    p95 = np.percentile(arr, 95)
    print(f"  {label}: p50={p50:.0f}ms  p95={p95:.0f}ms  max={arr.max():.0f}ms")


def main():
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY not set.")

    lf = init_langfuse()
    groq_client = Groq(api_key=groq_key)
    prompt_config = load_prompt_config()

    print("Loading resources...")
    chunks_map = load_chunks()
    bm25, chunk_ids = load_bm25()
    embed_model = SentenceTransformer(EMBEDDING_MODEL)
    reranker = CrossEncoder(RERANKER_MODEL)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = chroma_client.get_collection(COLLECTION_NAME)
    print("All resources loaded\n")

    test_queries = [
        "What is retrieval augmented generation?",
        "How does RAPTOR organize documents differently from flat chunking?",
        "How does self-RAG decide when to retrieve?",
        "What evaluation metrics does RAGAS use?",
        "How does ColBERT differ from dense passage retrieval?",
    ]

    all_results = []
    total_cost = 0

    for query in test_queries:
        print(f"Query: {query[:60]}...")
        result = rag_query(
            query, collection, embed_model, reranker,
            bm25, chunk_ids, chunks_map, groq_client, prompt_config, lf
        )
        all_results.append(result)
        total_cost += result["cost_usd"]
        lat = result["latencies_ms"]
        print(f"  embed={lat['embed_ms']}ms | bm25={lat['bm25_ms']}ms | "
              f"vector={lat['vector_ms']}ms | rerank={lat['rerank_ms']}ms | "
              f"llm={lat['llm_ms']}ms | total={lat['total_ms']}ms")
        print(f"  cost=${result['cost_usd']} | citations={result['citations']}")
        print()

    if lf:
        lf.flush()

    print("=" * 55)
    print("LATENCY SUMMARY (p50 / p95)")
    print("=" * 55)
    for stage in ["embed_ms", "bm25_ms", "vector_ms", "rerank_ms", "llm_ms", "total_ms"]:
        vals = [r["latencies_ms"][stage] for r in all_results]
        compute_percentiles(vals, stage.replace("_ms", ""))

    print(f"\nTotal cost for {len(test_queries)} queries: ${round(total_cost, 6)}")
    if lf:
        print("\nTraces sent to Langfuse Cloud -- check https://cloud.langfuse.com")


if __name__ == "__main__":
    main()
