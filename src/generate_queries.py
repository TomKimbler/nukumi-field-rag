import os
import csv
import json
import random
import re
import ollama
from datasets import load_from_disk
from pypdf import PdfReader
from tqdm import tqdm
from optiDataset import clean_pdf_text, iter_active_pdf_files, split_into_paragraphs

# Configuration
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
pdf_dir = os.path.join(project_root, "pdfFiles")
corpus_path = os.path.join(project_root, "search_corpus_dataset")
output_csv_path = os.path.join(project_root, "synthetic_queries.csv")
config_path = os.path.join(project_root, "config.json")

# The specific ollama model you have running (e.g., "gemma4:e2b", "llama3", "mistral", etc.)
OLLAMA_MODEL = "gemma4:e2b"


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


def generate_query_for_paragraph(paragraph: str, file_name: str) -> str | None:
    manual_title = os.path.splitext(file_name)[0]
    role = DOMAIN_CFG["domain_role"]
    equip_examples = DOMAIN_CFG["domain_equipment_examples"]
    good_ex = ", ".join(f'"{q}"' for q in DOMAIN_CFG["query_good_examples"])
    bad_ex = ", ".join(f'"{q}"' for q in DOMAIN_CFG["query_bad_examples"])

    prompt = f"""You are an expert {role}. 
Read the following paragraph extracted from a technical manual. 

Source Manual: {manual_title}

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

    if os.path.exists(corpus_path):
        print(f"Loading cleaned paragraphs from {corpus_path}...")
        dataset = load_from_disk(corpus_path)
        return [
            {"file_name": row["file_name"], "paragraph": row["paragraph"]}
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
        try:
            reader = PdfReader(pdf_path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    cleaned = clean_pdf_text(extracted)
                    paragraphs = split_into_paragraphs(cleaned, min_chars=150)
                    for para in paragraphs:
                        all_records.append({"file_name": filename, "paragraph": para})
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return all_records


def main():
    all_records = load_all_paragraphs()
    if not all_records:
        return

    random.seed(42)
    random.shuffle(all_records)

    print(f"Total valid paragraphs extracted: {len(all_records)}")

    sample_size = min(1500, len(all_records))
    records_to_process = all_records[:sample_size]

    print(f"Generating synthetic queries for {sample_size} paragraphs using {OLLAMA_MODEL}...")

    dataset_pairs = []
    for item in tqdm(records_to_process, desc="Generating Queries"):
        para = item["paragraph"]
        fname = item["file_name"]
        query = generate_query_for_paragraph(para, fname)

        # Allow natural conversational questions between 3 and 30 words
        if query and query.strip().upper() != "SKIP":
            word_count = len(query.split())
            if 3 <= word_count <= 30:
                dataset_pairs.append({
                    "query": query,
                    "paragraph": para,
                })

    print(f"Successfully generated {len(dataset_pairs)} high-quality training pairs!")
    with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "paragraph"])
        writer.writeheader()
        writer.writerows(dataset_pairs)

    print("Done! You now have a synthetic dataset for Contrastive Learning.")


if __name__ == "__main__":
    main()
