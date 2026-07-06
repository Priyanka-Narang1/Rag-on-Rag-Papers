"""
Fetches paper metadata from arXiv API to VERIFY each ID before downloading.
"""

import time
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

CANDIDATE_PAPERS = [
    ("2005.11401", "Retrieval-Augmented Generation"),
    ("2004.04906", "Dense Passage Retrieval"),
    ("2002.08909", "REALM"),
    ("2004.12832", "ColBERT"),
    ("2112.01488", "ColBERTv2"),
    ("2212.10496", "Zero-Shot Dense Retrieval"),
    ("2307.03172", "Lost in the Middle"),
    ("1908.10084", "Sentence-BERT"),
    ("2310.11511", "Self-RAG"),
    ("2401.15884", "Corrective Retrieval"),
    ("2403.10131", "RAFT"),
    ("2401.18059", "RAPTOR"),
    ("2404.16130", "Graph RAG"),
    ("2309.15217", "RAGAS"),
    ("2311.09476", "ARES"),
    ("2305.14283", "Query Rewriting"),
    ("2312.10997", "Retrieval-Augmented Generation for Large Language Models: A Survey"),
    ("2604.01733", "From BM25 to Corrective RAG"),
]

ARXIV_API = "http://export.arxiv.org/api/query?id_list={}"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw_pdfs"
RAW_DIR.mkdir(parents=True, exist_ok=True)
NS = {"atom": "http://www.w3.org/2005/Atom"}


def fetch_metadata(arxiv_id):
    url = ARXIV_API.format(arxiv_id)
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            xml_data = resp.read()
    except Exception as e:
        return {"error": str(e)}
    root = ET.fromstring(xml_data)
    entry = root.find("atom:entry", NS)
    if entry is None:
        return {"error": "no entry found"}
    raw_title = entry.find("atom:title", NS).text
    title = " ".join(raw_title.split())
    pdf_url = None
    for link in entry.findall("atom:link", NS):
        if link.get("title") == "pdf":
            pdf_url = link.get("href")
    if pdf_url is None:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    return {"title": title, "pdf_url": pdf_url}


def main():
    print(f"{'ID':<14}{'STATUS':<10}{'TITLE'}")
    print("-" * 90)
    verified = []
    failed = []
    for arxiv_id, expected_kw in CANDIDATE_PAPERS:
        meta = fetch_metadata(arxiv_id)
        time.sleep(1)
        if meta is None or "error" in meta:
            print(f"{arxiv_id:<14}{'FAIL':<10}{meta.get('error', 'unknown error')}")
            failed.append((arxiv_id, expected_kw))
            continue
        title = meta["title"]
        match = expected_kw.lower() in title.lower()
        status = "OK" if match else "MISMATCH"
        print(f"{arxiv_id:<14}{status:<10}{title}")
        if not match:
            print(f"   DEBUG repr(title) = {title!r}")
            print(f"   DEBUG repr(expected_kw) = {expected_kw!r}")
        if match:
            verified.append((arxiv_id, title, meta["pdf_url"]))
        else:
            failed.append((arxiv_id, expected_kw))
    print("\n" + "=" * 90)
    print(f"Verified: {len(verified)} / {len(CANDIDATE_PAPERS)}")
    if failed:
        print(f"Failed/mismatched: {[f[0] for f in failed]}")
    out_path = Path(__file__).parent.parent / "data" / "verified_papers.tsv"
    with open(out_path, "w") as f:
        f.write("arxiv_id\ttitle\tpdf_url\n")
        for arxiv_id, title, pdf_url in verified:
            f.write(f"{arxiv_id}\t{title}\t{pdf_url}\n")
    print(f"\nSaved verified list to {out_path}")


if __name__ == "__main__":
    main()
