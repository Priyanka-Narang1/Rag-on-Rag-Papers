"""
Embeds all 602 chunks using all-MiniLM-L6-v2 and stores in ChromaDB.
Run: python src/embed_chunks.py
"""

import json
import time
from pathlib import Path

from sentence_transformers import SentenceTransformer
import chromadb

DATA_DIR = Path(__file__).parent.parent / "data"
CHUNKS_FILE = DATA_DIR / "processed" / "chunks.jsonl"
CHROMA_DIR = DATA_DIR / "chroma_db"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "rag_papers"
BATCH_SIZE = 32


def load_chunks():
    chunks = []
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
    return chunks


def main():
    print(f"Loading chunks from {CHUNKS_FILE}...")
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Model loaded")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Delete collection if it exists so rerunning is safe
    existing = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing:
        client.delete_collection(COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}' (fresh run)")

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    print(f"Created ChromaDB collection: {COLLECTION_NAME}")

    # Embed and store in batches
    total_batches = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
    start_time = time.time()

    for batch_idx in range(total_batches):
        batch = chunks[batch_idx * BATCH_SIZE:(batch_idx + 1) * BATCH_SIZE]

        texts = [c["text"] for c in batch]
        ids = [c["chunk_id"] for c in batch]
        metadatas = [
            {
                "arxiv_id": c["arxiv_id"],
                "chunk_index": c["chunk_index"],
                "token_count": c["token_count"],
            }
            for c in batch
        ]

        embeddings = model.encode(texts, show_progress_bar=False).tolist()

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )

        print(f"  Batch {batch_idx + 1}/{total_batches} done "
              f"({min((batch_idx+1)*BATCH_SIZE, len(chunks))}/{len(chunks)} chunks)")

    elapsed = time.time() - start_time

    # Verify count matches what we put in
    stored_count = collection.count()
    print(f"\n{'='*50}")
    print(f"Embedding complete in {elapsed:.1f}s")
    print(f"Chunks embedded : {len(chunks)}")
    print(f"Stored in Chroma: {stored_count}")

    if stored_count != len(chunks):
        print(f"WARNING: count mismatch! Expected {len(chunks)}, got {stored_count}")
    else:
        print("Count verified: all chunks stored correctly")

    # Sanity check: run one test query
    print("\n--- Sanity check: test query ---")
    test_query = "What is retrieval augmented generation?"
    test_embedding = model.encode([test_query]).tolist()
    results = collection.query(
        query_embeddings=test_embedding,
        n_results=3,
        include=["documents", "metadatas", "distances"]
    )

    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        print(f"\n  Result {i+1}: [{meta['arxiv_id']}] distance={dist:.4f}")
        print(f"  {doc[:120].replace(chr(10), ' ')}...")

    print(f"\nChromaDB persisted at: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
