import re
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
VERIFIED_TSV = DATA_DIR / "verified_papers.tsv"

REFERENCE_SECTION_PATTERNS = [
    r"^\s*References\s*$",
    r"^\s*REFERENCES\s*$",
    r"^\s*Bibliography\s*$",
    r"^\s*BIBLIOGRAPHY\s*$",
    r"^\s*Works Cited\s*$",
]


def load_arxiv_ids():
    ids = []
    with open(VERIFIED_TSV, encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                ids.append(parts[0])
    return ids


def find_reference_section_start(text):
    lines = text.split("\n")

    # Pass 1: look for standard header patterns (searching backwards)
    for i in range(len(lines) - 1, max(len(lines) - 200, 0), -1):
        line = lines[i]
        for pattern in REFERENCE_SECTION_PATTERNS:
            if re.match(pattern, line):
                char_pos = sum(len(l) + 1 for l in lines[:i])
                return char_pos

    # Pass 2: detect numbered bibliography style [1] Author... appearing
    # consecutively in the last 30% of the document -- catches papers
    # like 2005.11401 that use numbered refs without a clear header line
    last_third_start = int(len(lines) * 0.70)
    consecutive_ref_lines = 0
    first_ref_line = None

    for i in range(last_third_start, len(lines)):
        line = lines[i].strip()
        if re.match(r"^\[\d+\]\s+[A-Z]", line):
            consecutive_ref_lines += 1
            if consecutive_ref_lines == 3:
                # 3 consecutive numbered ref lines = confident it's the bibliography
                first_ref_line = i - 2
                char_pos = sum(len(l) + 1 for l in lines[:first_ref_line])
                return char_pos
        else:
            consecutive_ref_lines = 0

    return len(text)


def fix_pdf_artifacts(text):
    text = re.sub(r"-\n(\w)", r"-\1", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def main():
    arxiv_ids = load_arxiv_ids()

    print(f"{'ID':<14}{'ORIGINAL CHARS':<18}{'AFTER REF REMOVAL':<20}{'REMOVED %'}")
    print("-" * 65)

    total_removed_pct = []

    for arxiv_id in arxiv_ids:
        txt_path = PROCESSED_DIR / f"{arxiv_id}.txt"
        if not txt_path.exists():
            print(f"{arxiv_id:<14}SKIP -- .txt not found")
            continue

        raw_text = txt_path.read_text(encoding="utf-8")
        original_len = len(raw_text)

        ref_start = find_reference_section_start(raw_text)
        text_no_refs = raw_text[:ref_start]
        clean_text = fix_pdf_artifacts(text_no_refs)
        cleaned_len = len(clean_text)

        removed_pct = (1 - cleaned_len / original_len) * 100
        total_removed_pct.append(removed_pct)

        print(f"{arxiv_id:<14}{original_len:<18}{cleaned_len:<20}{removed_pct:.1f}%")

        out_path = PROCESSED_DIR / f"{arxiv_id}_clean.txt"
        out_path.write_text(clean_text, encoding="utf-8")

    avg_removed = sum(total_removed_pct) / len(total_removed_pct)
    print("\n" + "=" * 65)
    print(f"Average content removed per paper: {avg_removed:.1f}%")
    print("Cleaned files saved to data/processed/*_clean.txt")


if __name__ == "__main__":
    main()
