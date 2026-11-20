"""
PDF Corpus Extraction and Cleaning Pipeline.

Extracts clean paragraphs with source manual filename and page metadata from pdfFiles/
(including subfolders, while skipping any subfolder or file starting with `ignore_` or `exclude_`)
and saves them to search_corpus_dataset/ for Hybrid Search (ModernBERT + BM25) and RAG.
"""

import os
import re
from datasets import Dataset
from pypdf import PdfReader

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

pdf_dir = os.path.join(project_root, "pdfFiles")
corpus_output_dir = os.path.join(project_root, "search_corpus_dataset")

EXCLUDED_PREFIXES = ("ignore_", "exclude_", ".")


def is_excluded_name(name: str) -> bool:
    """Return True if a directory or filename starts with 'ignore_', 'exclude_', or '.'."""
    return name.lower().startswith(EXCLUDED_PREFIXES)


def iter_active_pdf_files(pdf_directory: str) -> list[tuple[str, str]]:
    """
    Recursively walk `pdf_directory` and return a sorted list of `(filename, full_path)`
    for all active PDFs, pruning any subfolder or file starting with `ignore_` or `exclude_`.
    """
    if not os.path.exists(pdf_directory):
        return []

    active_pdfs: list[tuple[str, str]] = []
    for root, dirs, files in os.walk(pdf_directory):
        # Prune excluded subdirectories in-place so os.walk never descends into them
        dirs[:] = sorted(d for d in dirs if not is_excluded_name(d))
        for filename in sorted(files):
            if is_excluded_name(filename):
                continue
            if filename.lower().endswith(".pdf"):
                active_pdfs.append((filename, os.path.join(root, filename)))

    return active_pdfs


def find_active_pdf_path(pdf_directory: str, target_filename: str) -> str | None:
    """
    Locate `target_filename` inside `pdf_directory` or any of its active (non-excluded)
    subfolders. Returns None if not found or if located inside an `ignore_`/`exclude_` folder.
    """
    safe_target = os.path.basename(target_filename)
    if not safe_target or is_excluded_name(safe_target):
        return None

    for filename, full_path in iter_active_pdf_files(pdf_directory):
        if filename == safe_target:
            return full_path
    return None


def clean_pdf_text(text: str) -> str:
    """
    Clean common PDF extraction artifacts from technical HVAC/BMS manuals:
    - Collapse faux-bold duplicated characters (e.g., 'FFiigguurree 44' -> 'Figure 4',
      'CCoonnffiigg→→DDrriivveerrss' -> 'Config→Drivers') produced by overlapping PDF glyphs.
    - Fix broken hyphenated line wraps.
    """
    if not text:
        return ""

    def _dedup_match(match: re.Match) -> str:
        s = match.group(0)
        return s[::2]

    # Match sequences of 3+ doubled character pairs (6+ chars total, e.g., FFiigg...)
    text = re.sub(r"(?:([^\W\d_])\1){3,}", _dedup_match, text)
    # Also collapse doubled arrows or punctuation around faux-bold headers
    text = re.sub(r"([→•–—])\1+", r"\1", text)
    # Rejoin words hyphenated across line breaks (e.g. "cli-\nent" -> "client")
    text = re.sub(r"([a-zA-Z])-\s*\n\s*([a-z])", r"\1\2", text)
    return text


def split_into_paragraphs(text: str, min_chars: int = 120) -> list[str]:
    """Split cleaned page text into meaningful technical paragraphs for search indexing."""
    paragraphs = re.split(r"\n\s*\n", text)
    valid_paragraphs = []

    for p in paragraphs:
        p = p.strip()
        if len(p) < min_chars:
            continue
        # Filter out Tables of Contents (repeating dots like ".....")
        if re.search(r"\.{4,}", p):
            continue
        # Filter out blocks that are predominantly numbers/tables without prose
        numbers = sum(c.isdigit() for c in p)
        if numbers / len(p) > 0.30:
            continue

        p = re.sub(r"\s+", " ", p)
        valid_paragraphs.append(p)

    return valid_paragraphs


def generate_corpus_paragraphs(pdf_directory: str):
    """Yield clean paragraph-level records with source file_name and page metadata for search."""
    if not os.path.exists(pdf_directory):
        print(f"Error: Directory {pdf_directory} does not exist.")
        return

    active_pdfs = iter_active_pdf_files(pdf_directory)
    print(f"Found {len(active_pdfs)} active PDF manuals in {pdf_directory} (excluding ignore_/exclude_).")

    for filename, pdf_path in active_pdfs:
        try:
            reader = PdfReader(pdf_path)
            for page_idx, page in enumerate(reader.pages, start=1):
                extracted = page.extract_text()
                if not extracted:
                    continue
                cleaned = clean_pdf_text(extracted)
                for para in split_into_paragraphs(cleaned):
                    yield {
                        "file_name": filename,
                        "page": page_idx,
                        "paragraph": para,
                    }
        except Exception as e:
            print(f"Error reading {filename}: {e}")


def main():
    print(f"Building clean paragraph search corpus from {pdf_dir}...")
    corpus_dataset = Dataset.from_generator(
        generate_corpus_paragraphs,
        gen_kwargs={"pdf_directory": pdf_dir},
    )
    corpus_dataset.save_to_disk(corpus_output_dir)
    print(f"Search corpus ({len(corpus_dataset)} paragraphs) saved to {corpus_output_dir}")


if __name__ == "__main__":
    main()
