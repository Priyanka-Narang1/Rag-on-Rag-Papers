"""
Downloads the verified PDFs listed in data/verified_papers.tsv into data/raw_pdfs/.
Run AFTER fetch_papers.py has produced a clean verified_papers.tsv.

This script does NOT extract text -- that's a separate step (extract_text.py)
so download failures and extraction failures don't get tangled together.

Run: python src/download_pdfs.py
"""

import csv
import time
import urllib.request
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw_pdfs"
VERIFIED_TSV = DATA_DIR / "verified_papers.tsv"

RAW_DIR.mkdir(parents=True, exist_ok=True)


def load_verified_papers():
    papers = []
    with open(VERIFIED_TSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            papers.append(row)
    return papers


def download_pdf(arxiv_id, pdf_url, dest_path):
    if dest_path.exists() and dest_path.stat().st_size > 0:
        return {"status": "SKIPPED (already exists)", "size_kb": dest_path.stat().st_size // 1024}

    url = pdf_url
    if not url.endswith(".pdf") and "/pdf/" not in url:
        url = f"https://arxiv.org/pdf/{arxiv_id}"

    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
    except Exception as e:
        return {"status": f"FAIL ({e})", "size_kb": 0}

    if len(data) < 1000:
        return {"status": "FAIL (response too small, likely not a valid PDF)", "size_kb": len(data) // 1024}

    dest_path.write_bytes(data)
    return {"status": "OK", "size_kb": len(data) // 1024}


def main():
    papers = load_verified_papers()
    print(f"Loaded {len(papers)} verified papers from {VERIFIED_TSV}")
    print(f"{'ID':<14}{'STATUS':<35}{'SIZE'}")
    print("-" * 70)

    results = []
    for row in papers:
        arxiv_id = row["arxiv_id"]
        pdf_url = row["pdf_url"]
        dest_path = RAW_DIR / f"{arxiv_id}.pdf"

        result = download_pdf(arxiv_id, pdf_url, dest_path)
        print(f"{arxiv_id:<14}{result['status']:<35}{result['size_kb']} KB")
        results.append((arxiv_id, result["status"]))

        time.sleep(2)

    ok_count = sum(1 for _, s in results if s == "OK" or "SKIPPED" in s)
    print("\n" + "=" * 70)
    print(f"Downloaded/present: {ok_count} / {len(papers)}")

    failed = [aid for aid, s in results if "FAIL" in s]
    if failed:
        print(f"FAILED: {failed}")
        print("-> Rerun this script; already-downloaded files are skipped automatically.")
    else:
        print("All PDFs downloaded successfully.")


if __name__ == "__main__":
    main()