import argparse
from collections import defaultdict
import os
import csv
import json
import random
import re
import ollama
from datasets import load_from_disk
from pypdf import PdfReader
from tqdm import tqdm
from optiDataset import (
    build_active_pdf_category_map,
    clean_pdf_text,
    format_category_from_path,
    iter_active_pdf_files,
    split_into_paragraphs,
)

# Configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
pdf_dir = os.path.join(project_root, "pdfFiles")
corpus_path = os.path.join(project_root, "search_corpus_dataset")
output_csv_path = os.path.join(project_root, "synthetic_queries.csv")
config_path = os.path.join(project_root, "config.json")
modelfile_path = os.path.join(project_root, "Modelfile")


def load_ollama_model_name() -> str:
    """Read the Ollama model from OLLAMA_MODEL env var or the FROM line in Modelfile."""
    env_model = os.environ.get("OLLAMA_MODEL", "").strip()
    if env_model:
        return env_model
    if os.path.exists(modelfile_path):
        try:
            with open(modelfile_path, "r", encoding="utf-8") as f:
                match = re.search(r"^FROM\s+(\S+)", f.read(), re.MULTILINE)
                if match:
                    return match.group(1).strip()
        except Exception:
            pass
    return "gemma4:e2b"


# Automatically uses the FROM model in Modelfile (or OLLAMA_MODEL env var)
OLLAMA_MODEL = load_ollama_model_name()


def load_domain_config() -> dict:
    defaults = {
        "domain_role": "HVAC and BMS controls technician",
        "domain_equipment_examples": "TC500A, TC300A, CIPer 30, Eaglehawk, Spyder, WEB-8000, Optimizer",
        "query_good_examples": [
            "How do I set the baud rate on a BACnet MSTP port?",
            "What is the default IP address for the Web-8000?",
            "Where can I find the wiring diagram for the CIPer 30?",
        ],
        "query_bad_examples": [
            "BACnet MSTP baud rate",
            "CIPer 30 wiring",
        ],
    }
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                defaults.update(json.load(f))
        except Exception as e:
            print(f"Warning: Could not parse {config_path} ({e}), using defaults.")
    return defaults


DOMAIN_CFG = load_domain_config()


def generate_query_for_paragraph(paragraph: str, file_name: str, category: str = "") -> str | None:
    manual_title = os.path.splitext(file_name)[0]
    role = DOMAIN_CFG["domain_role"]
    equip_examples = DOMAIN_CFG["domain_equipment_examples"]
    good_ex = ", ".join(f'"{q}"' for q in DOMAIN_CFG["query_good_examples"])
    bad_ex = ", ".join(f'"{q}"' for q in DOMAIN_CFG["query_bad_examples"])
    category_line = f"Equipment / Trade Category: {category}\n" if category else ""

    prompt = f"""You are an expert {role}. 
Read the following paragraph extracted from a technical manual. 

{category_line}Source Manual: {manual_title}

Your task: Write ONE realistic, conversational question that a technician would ask a support chatbot to find this exact information.

RULES:
1. You MUST write it as a natural, conversational question (e.g., "How do I...", "What is the procedure for...", "Can you tell me...").
2. Good Examples: {good_ex}
3. Bad Examples: {bad_ex} (These are search keywords, we want full questions).
4. SPECIFICITY RULE: If the Source Manual or paragraph is about a specific product/controller model (e.g., {equip_examples}), you MUST include that specific model name in the question instead of saying "this unit" or a generic category name. Only write a generic equipment question if the paragraph is teaching a universal domain principle.
5. If the paragraph is gibberish, a table of contents, legal boilerplate, or in a foreign language (e.g., German, French, Spanish), reply EXACTLY with the word: SKIP
6. Output ONLY the question. No quotes, no intro text.

Paragraph:
{paragraph}

Search Query:"""

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response["message"]["content"].strip(" \"'")
    except Exception as e:
        print(f"Error calling Ollama: {e}")
        return None


def load_all_paragraphs() -> list[dict[str, str]]:
    """
    Load cleaned paragraphs with their source file_name from search_corpus_dataset or raw PDFs,
    automatically excluding any PDF inside an `ignore_` or `exclude_` folder.
    """
    active_pdfs = iter_active_pdf_files(pdf_dir)
    active_names = {fname for fname, _ in active_pdfs} if active_pdfs else None
    live_cat_map = build_active_pdf_category_map(pdf_dir) if active_pdfs else {}

    if os.path.exists(corpus_path):
        print(f"Loading cleaned paragraphs from {corpus_path}...")
        dataset = load_from_disk(corpus_path)
        has_cat = "category" in dataset.column_names
        return [
            {
                "file_name": row["file_name"],
                "category": live_cat_map.get(row["file_name"], row["category"] if has_cat else ""),
                "paragraph": row["paragraph"],
            }
            for row in dataset
            if len(row["paragraph"]) >= 150
            and (active_names is None or row["file_name"] in active_names)
        ]

    print("Extracting and cleaning paragraphs from raw PDFs...")
    if not active_pdfs:
        print(f"Error: No active PDFs found in {pdf_dir}.")
        return []

    all_records = []
    shuffled_pdfs = list(active_pdfs)
    random.seed(42)
    random.shuffle(shuffled_pdfs)

    for filename, pdf_path in shuffled_pdfs:
        category = format_category_from_path(pdf_dir, pdf_path)
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    cleaned = clean_pdf_text(extracted)
                    paragraphs = split_into_paragraphs(cleaned, min_chars=150)
                    for para in paragraphs:
                        all_records.append({
                            "file_name": filename,
                            "category": category,
                            "paragraph": para,
                        })
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return all_records


def stratified_sample_records(
    all_records: list[dict[str, str]],
    max_samples: int,
    seed: int = 42,
) -> list[dict[str, str]]:
    """
    Round-robin stratified sampling across manuals so every PDF gets balanced coverage
    instead of a handful of 500-page manuals dominating the training set.
    If max_samples <= 0 or >= len(all_records), returns all records in balanced manual order.
    """
    rng = random.Random(seed)
    by_manual: dict[str, list[dict[str, str]]] = defaultdict(list)
    for rec in all_records:
        by_manual[rec["file_name"]].append(rec)

    manual_names = sorted(by_manual.keys())
    rng.shuffle(manual_names)
    for fname in manual_names:
        rng.shuffle(by_manual[fname])

    ordered: list[dict[str, str]] = []
    idx = 0
    while True:
        added_any = False
        for fname in manual_names:
            bucket = by_manual[fname]
            if idx < len(bucket):
                ordered.append(bucket[idx])
                added_any = True
        if not added_any:
            break
        idx += 1

    if max_samples > 0 and max_samples < len(ordered):
        return ordered[:max_samples]
    return ordered


def load_existing_paragraphs(csv_path: str) -> set[str]:
    """Load already-processed paragraphs from an existing synthetic_queries.csv for --resume."""
    seen: set[str] = set()
    if not os.path.exists(csv_path):
        return seen
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            para = (row.get("paragraph") or "").strip()
            if para:
                seen.add(para)
    return seen


def main():
    global OLLAMA_MODEL
    parser = argparse.ArgumentParser(
        description="Generate synthetic technician questions paired with manual paragraphs."
    )
    default_max = int(os.environ.get("MAX_QUERY_SAMPLES", "4000"))
    parser.add_argument(
        "--max-samples",
        type=int,
        default=default_max,
        help=f"Maximum paragraphs to process via stratified manual sampling (default: {default_max}; 0 = all).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=OLLAMA_MODEL,
        help=f"Ollama model to use for query generation (default: {OLLAMA_MODEL}).",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume and append to existing synthetic_queries.csv, skipping already-processed paragraphs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible stratified sampling (default: 42).",
    )
    args = parser.parse_args()

    OLLAMA_MODEL = args.model

    all_records = load_all_paragraphs()
    if not all_records:
        return

    num_manuals = len({r["file_name"] for r in all_records})
    print(f"Total valid paragraphs extracted: {len(all_records)} across {num_manuals} active manuals.")

    records_to_process = stratified_sample_records(all_records, args.max_samples, seed=args.seed)

    existing_paragraphs: set[str] = set()
    if args.resume and os.path.exists(output_csv_path):
        existing_paragraphs = load_existing_paragraphs(output_csv_path)
        before = len(records_to_process)
        records_to_process = [r for r in records_to_process if r["paragraph"].strip() not in existing_paragraphs]
        print(f"Resuming: skipping {before - len(records_to_process)} already-generated paragraphs.")

    if not records_to_process:
        print("Nothing new to generate!")
        return

    sampled_manuals = len({r["file_name"] for r in records_to_process})
    print(
        f"Generating synthetic queries for {len(records_to_process)} paragraphs "
        f"(covering {sampled_manuals} manuals) using {OLLAMA_MODEL}..."
    )

    write_mode = "a" if (args.resume and os.path.exists(output_csv_path) and os.path.getsize(output_csv_path) > 0) else "w"
    fieldnames = ["query", "paragraph", "file_name", "category"]
    if write_mode == "a":
        with open(output_csv_path, "r", encoding="utf-8") as existing_f:
            existing_reader = csv.reader(existing_f)
            existing_header = next(existing_reader, None)
            if existing_header:
                fieldnames = existing_header

    generated_count = 0

    # Stream rows to disk incrementally so interrupted runs preserve progress
    with open(output_csv_path, write_mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_mode == "w":
            writer.writeheader()

        for item in tqdm(records_to_process, desc="Generating Queries"):
            para = item["paragraph"]
            fname = item["file_name"]
            cat = item.get("category", "")
            query = generate_query_for_paragraph(para, fname, category=cat)

            # Allow natural conversational questions between 3 and 30 words
            if query and query.strip().upper() != "SKIP":
                word_count = len(query.split())
                if 3 <= word_count <= 30:
                    writer.writerow({
                        "query": query,
                        "paragraph": para,
                        "file_name": fname,
                        "category": cat,
                    })
                    f.flush()
                    generated_count += 1

    print(f"Successfully generated {generated_count} high-quality training pairs -> {output_csv_path}")
    print("Done! You now have a stratified synthetic dataset for Contrastive Learning.")


if __name__ == "__main__":
    main()
