import csv
from pathlib import Path

import fitz

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw_pdfs"
PROCESSED_DIR = DATA_DIR / "processed"
VERIFIED_TSV = DATA_DIR / "verified_papers.tsv"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

MIN_EXPECTED_CHARS = 2000


def load_verified_papers():
    papers = []
    with open(VERIFIED_TSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            papers.append(row)
    return papers


def extract_pdf_text(pdf_path):
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return {"status": f"FAIL (could not open: {e})", "char_count": 0, "page_count": 0, "text": ""}

    page_texts = []
    for page in doc:
        page_text = page.get_text("text", sort=True)
        page_texts.append(page_text)
    doc.close()

    full_text = "\n\n".join(page_texts)
    char_count = len(full_text.strip())
    page_count = len(page_texts)

    if char_count < MIN_EXPECTED_CHARS:
        status = f"WARNING (only {char_count} chars across {page_count} pages -- check manually)"
    else:
        status = "OK"

    return {"status": status, "char_count": char_count, "page_count": page_count, "text": full_text}


def main():
    papers = load_verified_papers()
    print(f"{'ID':<14}{'STATUS':<55}{'PAGES':<7}{'CHARS'}")
    print("-" * 95)

    results = []
    for row in papers:
        arxiv_id = row["arxiv_id"]
        pdf_path = RAW_DIR / f"{arxiv_id}.pdf"

        if not pdf_path.exists():
            print(f"{arxiv_id:<14}{'FAIL (PDF not found, run download_pdfs.py first)':<55}")
            results.append((arxiv_id, "FAIL (missing PDF)"))
            continue

        result = extract_pdf_text(pdf_path)
        print(f"{arxiv_id:<14}{result['status']:<55}{result['page_count']:<7}{result['char_count']}")
        results.append((arxiv_id, result["status"]))

        if result["text"]:
            out_path = PROCESSED_DIR / f"{arxiv_id}.txt"
            out_path.write_text(result["text"], encoding="utf-8")

    ok_count = sum(1 for _, s in results if s == "OK")
    warn_count = sum(1 for _, s in results if "WARNING" in s)
    fail_count = sum(1 for _, s in results if "FAIL" in s)

    print("\n" + "=" * 95)
    print(f"OK: {ok_count} | WARNINGS: {warn_count} | FAILED: {fail_count} (out of {len(papers)})")

    if warn_count or fail_count:
        print("-> Review flagged papers manually before moving to chunking.")
        flagged = [aid for aid, s in results if "WARNING" in s or "FAIL" in s]
        print(f"Flagged IDs: {flagged}")
    else:
        print("All papers extracted cleanly. Ready for chunking.")


if __name__ == "__main__":
    main()
