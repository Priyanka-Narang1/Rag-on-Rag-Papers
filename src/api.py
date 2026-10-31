import json
import os
import time
import pickle
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb
from groq import Groq
import yaml

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
CHUNKS_FILE = DATA_DIR / "processed" / "chunks.jsonl"
BM25_INDEX_FILE = DATA_DIR / "processed" / "bm25_index.pkl"
CHROMA_DIR = DATA_DIR / "chroma_db"
VERIFIED_TSV = DATA_DIR / "verified_papers.tsv"
PROMPTS_DIR = BASE_DIR / "prompts"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
COLLECTION_NAME = "rag_papers"
GROQ_MODEL = "llama-3.1-8b-instant"
RRF_K = 60
RERANK_POOL = 30
FINAL_TOP_K = 5
COST_PER_1K_INPUT = 0.00005
COST_PER_1K_OUTPUT = 0.00008

PAPER_META = {
    "2005.11401": {"authors": "Lewis et al.", "year": 2020, "abstract": "Introduces RAG, combining parametric and non-parametric memory for knowledge-intensive NLP tasks."},
    "2004.04906": {"authors": "Karpukhin et al.", "year": 2020, "abstract": "Dense Passage Retrieval uses dual-encoder BERT models for open-domain QA, outperforming BM25."},
    "2002.08909": {"authors": "Guu et al.", "year": 2020, "abstract": "REALM augments language model pre-training with a learned knowledge retriever over Wikipedia."},
    "2004.12832": {"authors": "Khattab & Zaharia", "year": 2020, "abstract": "ColBERT introduces late interaction over BERT for efficient passage search via MaxSim operator."},
    "2112.01488": {"authors": "Santhanam et al.", "year": 2021, "abstract": "ColBERTv2 achieves state-of-the-art retrieval with aggressive residual compression."},
    "2212.10496": {"authors": "Gao et al.", "year": 2022, "abstract": "HyDE generates hypothetical documents via LLM to bridge query and document embedding spaces."},
    "2307.03172": {"authors": "Liu et al.", "year": 2023, "abstract": "Language models perform best when relevant information appears at the beginning or end of long contexts."},
    "1908.10084": {"authors": "Reimers & Gurevych", "year": 2019, "abstract": "Sentence-BERT produces fixed-size sentence embeddings using siamese BERT networks."},
    "2310.11511": {"authors": "Asai et al.", "year": 2023, "abstract": "Self-RAG trains models to retrieve, generate, and critique outputs using special reflection tokens."},
    "2401.15884": {"authors": "Yan et al.", "year": 2024, "abstract": "Corrective RAG evaluates retrieved documents and triggers web search when retrieval quality is low."},
    "2403.10131": {"authors": "Zhang et al.", "year": 2024, "abstract": "RAFT fine-tunes LLMs to answer from relevant documents while ignoring distractors."},
    "2401.18059": {"authors": "Sarthi et al.", "year": 2024, "abstract": "RAPTOR recursively embeds, clusters, and summarizes chunks into a tree for multi-level retrieval."},
    "2404.16130": {"authors": "Edge et al.", "year": 2024, "abstract": "GraphRAG builds a knowledge graph with community detection for global query-focused summarization."},
    "2309.15217": {"authors": "Es et al.", "year": 2023, "abstract": "RAGAS provides reference-free automated evaluation using faithfulness, answer relevancy, and context relevance."},
    "2311.09476": {"authors": "Saad-Falcon et al.", "year": 2023, "abstract": "ARES trains lightweight LLM judges using synthetic data for accurate RAG evaluation."},
    "2305.14283": {"authors": "Ma et al.", "year": 2023, "abstract": "Rewrite-Retrieve-Read uses a trainable rewriter to reformulate queries before retrieval."},
    "2312.10997": {"authors": "Gao et al.", "year": 2023, "abstract": "Comprehensive survey of RAG covering Naive, Advanced, and Modular RAG paradigms."},
    "2604.01733": {"authors": "Akarsu et al.", "year": 2026, "abstract": "Benchmarks ten retrieval strategies showing hybrid fusion consistently outperforms dense retrieval alone."},
}

app = FastAPI(title="RAG Research Studio API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

resources = {}


@app.on_event("startup")
async def load_resources():
    print("Loading RAG pipeline resources...")
    chunks_map = {}
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            chunks_map[c["chunk_id"]] = c
    resources["chunks_map"] = chunks_map

    with open(BM25_INDEX_FILE, "rb") as f:
        payload = pickle.load(f)
    resources["bm25"] = payload["bm25"]
    resources["chunk_ids"] = payload["chunk_ids"]

    resources["embed_model"] = SentenceTransformer(EMBEDDING_MODEL)
    resources["reranker"] = CrossEncoder(RERANKER_MODEL)

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    resources["collection"] = client.get_collection(COLLECTION_NAME)

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY not set")
    resources["groq_client"] = Groq(api_key=api_key)

    with open(PROMPTS_DIR / "v2.yaml", encoding="utf-8") as f:
        resources["prompt_config"] = yaml.safe_load(f)

    print("All resources loaded.")


class QueryRequest(BaseModel):
    question: str


class CitationOut(BaseModel):
    chunk_id: str
    arxiv_id: str
    text_preview: str
    paper_title: str
    authors: str
    year: int


class QueryResponse(BaseModel):
    answer: str
    citations: list[CitationOut]
    hallucinated_citations: list[str]
    latencies_ms: dict
    tokens: dict
    cost_usd: float
    prompt_version: str


class PaperOut(BaseModel):
    arxiv_id: str
    title: str
    authors: str
    year: int
    abstract: str
    pdf_url: str


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


@app.get("/api/health")
def health():
    return {"status": "ok", "model": GROQ_MODEL}


@app.get("/api/papers", response_model=list[PaperOut])
def get_papers():
    papers = []
    with open(VERIFIED_TSV, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            arxiv_id, title = parts[0], parts[1]
            meta = PAPER_META.get(arxiv_id, {})
            papers.append(PaperOut(
                arxiv_id=arxiv_id,
                title=title,
                authors=meta.get("authors", "Unknown"),
                year=meta.get("year", 0),
                abstract=meta.get("abstract", ""),
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}",
            ))
    return papers


@app.post("/api/query", response_model=QueryResponse)
def query_rag(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    chunks_map = resources["chunks_map"]
    bm25 = resources["bm25"]
    chunk_ids = resources["chunk_ids"]
    embed_model = resources["embed_model"]
    reranker = resources["reranker"]
    collection = resources["collection"]
    groq_client = resources["groq_client"]
    prompt_config = resources["prompt_config"]

    latencies = {}
    total_start = time.perf_counter()

    t0 = time.perf_counter()
    query_embedding = embed_model.encode([req.question]).tolist()
    latencies["embed_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    t0 = time.perf_counter()
    vr = collection.query(query_embeddings=query_embedding,
                          n_results=RERANK_POOL, include=["distances"])
    vector_results = [{"chunk_id": cid} for cid in vr["ids"][0]]
    latencies["vector_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    t0 = time.perf_counter()
    bm25_results = bm25_search(bm25, chunk_ids, req.question, top_k=RERANK_POOL)
    latencies["bm25_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    fused = reciprocal_rank_fusion(vector_results, bm25_results)
    candidate_ids = [cid for cid, _ in fused[:RERANK_POOL]]

    t0 = time.perf_counter()
    top_chunks = rerank(req.question, candidate_ids, chunks_map, reranker, top_k=FINAL_TOP_K)
    latencies["rerank_ms"] = round((time.perf_counter() - t0) * 1000, 2)

    retrieved_ids = [c["chunk_id"] for c in top_chunks]

    context = "\n\n---\n\n".join(
        f"[{c['chunk_id']}]\n{c['text']}" for c in top_chunks
    )
    system = prompt_config.get("system_prompt", "")
    user = f"Context:\n{context}\n\nQuestion: {req.question}"

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
    latencies["total_ms"] = round((time.perf_counter() - total_start) * 1000, 2)

    raw = response.choices[0].message.content
    usage = response.usage
    cost_usd = round(
        (usage.prompt_tokens / 1000 * COST_PER_1K_INPUT) +
        (usage.completion_tokens / 1000 * COST_PER_1K_OUTPUT), 6
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

    citation_out = []
    paper_titles = {}
    with open(VERIFIED_TSV, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                paper_titles[parts[0]] = parts[1]

    for cid in valid_citations:
        chunk = chunks_map.get(cid, {})
        arxiv_id = chunk.get("arxiv_id", "")
        meta = PAPER_META.get(arxiv_id, {})
        citation_out.append(CitationOut(
            chunk_id=cid,
            arxiv_id=arxiv_id,
            text_preview=chunk.get("text", "")[:200],
            paper_title=paper_titles.get(arxiv_id, ""),
            authors=meta.get("authors", ""),
            year=meta.get("year", 0),
        ))

    return QueryResponse(
        answer=answer,
        citations=citation_out,
        hallucinated_citations=hallucinated,
        latencies_ms=latencies,
        tokens={"prompt": usage.prompt_tokens, "completion": usage.completion_tokens},
        cost_usd=cost_usd,
        prompt_version=str(prompt_config.get("version", "1.0")),
    )

