import json
import re
from pathlib import Path
import tiktoken

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
VERIFIED_TSV = DATA_DIR / "verified_papers.tsv"
CHUNKS_OUT = PROCESSED_DIR / "chunks.jsonl"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
ENCODING_NAME = "cl100k_base"


def load_arxiv_ids():
    ids = []
    with open(VERIFIED_TSV, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                ids.append(parts[0])
    return ids


def chunk_tokens(text, enc, chunk_size, overlap):
    tokens = enc.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens_slice = tokens[start:end]
        chunk_text = enc.decode(chunk_tokens_slice)
        chunks.append({"tokens": chunk_tokens_slice, "text": chunk_text})
        if end == len(tokens):
            break
        start += chunk_size - overlap
    return chunks


def main():
    enc = tiktoken.get_encoding(ENCODING_NAME)
    arxiv_ids = load_arxiv_ids()

    total_chunks = 0
    per_paper_stats = []

    with open(CHUNKS_OUT, "w", encoding="utf-8") as out_f:
        for arxiv_id in arxiv_ids:
            # Use _clean.txt (reference-section removed + artifact-fixed)
            txt_path = PROCESSED_DIR / f"{arxiv_id}_clean.txt"
            if not txt_path.exists():
                print(f"SKIP {arxiv_id} -- _clean.txt not found, run clean_texts.py first")
                continue

            raw_text = txt_path.read_text(encoding="utf-8")
            chunks = chunk_tokens(raw_text, enc, CHUNK_SIZE, CHUNK_OVERLAP)

            for i, chunk in enumerate(chunks):
                record = {
                    "chunk_id": f"{arxiv_id}_chunk_{i:04d}",
                    "arxiv_id": arxiv_id,
                    "chunk_index": i,
                    "token_count": len(chunk["tokens"]),
                    "text": chunk["text"],
                }
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            per_paper_stats.append((arxiv_id, len(chunks)))
            total_chunks += len(chunks)

    print(f"{'ID':<14}{'CHUNKS'}")
    print("-" * 25)
    for arxiv_id, n in per_paper_stats:
        print(f"{arxiv_id:<14}{n}")

    print("\n" + "=" * 25)
    print(f"Total chunks: {total_chunks}")
    print(f"Saved to: {CHUNKS_OUT}")

    print("\n--- Sample chunks (sanity check) ---")
    with open(CHUNKS_OUT, encoding="utf-8") as f:
        lines = f.readlines()

    import random
    random.seed(42)
    samples = random.sample(lines, min(5, len(lines)))
    for line in samples:
        c = json.loads(line)
        preview = c["text"][:120].replace("\n", " ")
        print(f"  [{c['chunk_id']}] tokens={c['token_count']} | {preview}...")


if __name__ == "__main__":
    main()
