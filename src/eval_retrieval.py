"""
Retrieval Evaluation Harness for Nukumi (ModernBERT Semantic vs. BM25 vs. Hybrid RRF).

Measures Recall@1, Recall@4, and MRR@4 so you can objectively verify whether changes to
chunking, synthetic query generation, RRF weights, or train_st.py actually improve retrieval.

Usage:
  python src/eval_retrieval.py                      # Evaluate using eval_queries.json
  python src/eval_retrieval.py --from-feedback      # Evaluate using 👍 rows in feedback_log.csv
  python src/eval_retrieval.py --top-k 4 --verbose  # Show every query's top hits & misses
"""

import argparse
import csv
import json
import os
import sys
import torch
from sentence_transformers import SentenceTransformer, util

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from test_search import (
    BM25Index,
    filter_active_corpus,
    hybrid_search,
    load_or_build_corpus,
    load_or_compute_embeddings,
    model_path,
    resolve_corpus_categories,
)

DEFAULT_EVAL_FILE = os.path.join(project_root, "eval_queries.json")
FEEDBACK_CSV_PATH = os.path.join(project_root, "feedback_log.csv")


def load_eval_cases(eval_file: str, from_feedback: bool) -> list[dict]:
    cases: list[dict] = []

    if from_feedback:
        if not os.path.exists(FEEDBACK_CSV_PATH):
            print(f"Error: {FEEDBACK_CSV_PATH} not found.")
            return []
        with open(FEEDBACK_CSV_PATH, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                rating = (row.get("rating") or "").strip().lower()
                fname = (row.get("file_name") or "").strip()
                if rating == "up" and fname and fname != "Standard Engineering Formula":
                    cases.append({
                        "query": (row.get("user_query") or row.get("search_query") or "").strip(),
                        "expected_file": fname,
                        "expected_page": int(row["page"]) if str(row.get("page", "")).isdigit() else None,
                    })
        return [c for c in cases if c["query"]]

    if not os.path.exists(eval_file):
        print(f"Error: Evaluation file not found at {eval_file}.")
        return []

    with open(eval_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        print(f"Error: {eval_file} must contain a JSON list of evaluation cases.")
        return []
    return [c for c in data if isinstance(c, dict) and c.get("query") and c.get("expected_file")]


def is_hit_match(case: dict, file_name: str, page: int) -> bool:
    """
    Match if `expected_file` (case-insensitive substring or exact filename) matches `file_name`,
    and if `expected_page` is specified, matches within ±1 page (to tolerate PDF page offsets).
    """
    expected_file = case["expected_file"].lower()
    if expected_file not in file_name.lower():
        return False
    expected_page = case.get("expected_page")
    if expected_page is not None:
        return abs(int(page) - int(expected_page)) <= 1
    return True


def score_ranking(
    case: dict,
    ranked_ids: list[int],
    file_names: list[str],
    pages: list[int],
    top_k: int,
) -> tuple[bool, bool, float]:
    """Return (hit_at_1, hit_at_k, reciprocal_rank)."""
    for rank, cid in enumerate(ranked_ids[:top_k], start=1):
        if is_hit_match(case, file_names[cid], pages[cid]):
            return (rank == 1, True, 1.0 / rank)
    return (False, False, 0.0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Nukumi Hybrid / Semantic / BM25 retrieval.")
    parser.add_argument("--eval-file", default=DEFAULT_EVAL_FILE, help="Path to golden JSON eval file.")
    parser.add_argument("--from-feedback", action="store_true", help="Evaluate on 👍 rows from feedback_log.csv.")
    parser.add_argument("--top-k", type=int, default=4, help="Cutoff K for Recall@K and MRR@K (default: 4).")
    parser.add_argument("--verbose", action="store_true", help="Print per-query pass/miss details.")
    args = parser.parse_args()

    cases = load_eval_cases(args.eval_file, args.from_feedback)
    if not cases:
        print("No evaluation cases loaded.")
        return

    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Run train_st.py first.")
        return

    corpus_dataset = load_or_build_corpus()
    if corpus_dataset is None or len(corpus_dataset) == 0:
        print("Error: Search corpus is empty.")
        return

    retriever_device = "cpu" if os.environ.get("CPU_RETRIEVER", "0") == "1" else None
    model = SentenceTransformer(model_path, device=retriever_device)

    paragraphs = list(corpus_dataset["paragraph"])
    file_names = list(corpus_dataset["file_name"])
    pages = list(corpus_dataset["page"])
    raw_cats = list(corpus_dataset["category"]) if "category" in corpus_dataset.column_names else None
    categories = resolve_corpus_categories(file_names, raw_cats)

    paragraph_embeddings = load_or_compute_embeddings(
        model, paragraphs, file_names=file_names, categories=categories
    )
    if model.device.type == "cpu":
        paragraph_embeddings = paragraph_embeddings.cpu()
    paragraphs, file_names, pages, paragraph_embeddings, categories = filter_active_corpus(
        paragraphs, file_names, pages, paragraph_embeddings, categories=categories
    )
    bm25_index = BM25Index(paragraphs, file_names=file_names, categories=categories)

    metrics = {
        "semantic": {"r1": 0, "rk": 0, "mrr": 0.0},
        "bm25": {"r1": 0, "rk": 0, "mrr": 0.0},
        "hybrid": {"r1": 0, "rk": 0, "mrr": 0.0},
    }
    misses: list[str] = []

    for idx, case in enumerate(cases, start=1):
        query = case["query"]

        # Semantic-only
        q_emb = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
        sem_hits = util.semantic_search(q_emb, paragraph_embeddings, top_k=args.top_k)[0]
        sem_ids = [int(h["corpus_id"]) for h in sem_hits]

        # BM25-only
        bm25_hits = bm25_index.search(query, top_k=args.top_k)
        bm25_ids = [int(h["corpus_id"]) for h in bm25_hits]

        # Hybrid RRF
        hyb_hits = hybrid_search(
            query=query,
            model=model,
            paragraph_embeddings=paragraph_embeddings,
            bm25_index=bm25_index,
            top_k=args.top_k,
        )
        hyb_ids = [int(h["corpus_id"]) for h in hyb_hits]

        for name, ids in (("semantic", sem_ids), ("bm25", bm25_ids), ("hybrid", hyb_ids)):
            r1, rk, rr = score_ranking(case, ids, file_names, pages, args.top_k)
            metrics[name]["r1"] += int(r1)
            metrics[name]["rk"] += int(rk)
            metrics[name]["mrr"] += rr

        hyb_r1, hyb_rk, _ = score_ranking(case, hyb_ids, file_names, pages, args.top_k)
        top1_desc = (
            f"{file_names[hyb_ids[0]]} (p.{pages[hyb_ids[0]]})"
            if hyb_ids
            else "no hits"
        )
        exp_desc = case["expected_file"] + (f" (p.{case['expected_page']})" if case.get("expected_page") else "")

        if not hyb_rk:
            misses.append(f"  [MISS #{idx}] Q: {query}\n    Expected: {exp_desc}\n    Got #1  : {top1_desc}")
        elif args.verbose:
            badge = "PASS@1" if hyb_r1 else f"PASS@{args.top_k}"
            print(f"  [{badge} #{idx}] Q: {query} -> {top1_desc}")

    total = len(cases)
    k = args.top_k
    print("\n" + "=" * 68)
    print(f"📊 NUKUMI RETRIEVAL EVALUATION ({total} queries | {len(paragraphs)} chunks)")
    print("=" * 68)
    print(f"{'Engine':<16} {'Recall@1':>12} {f'Recall@{k}':>12} {f'MRR@{k}':>12}")
    print("-" * 68)
    for label, key in (("ModernBERT Only", "semantic"), ("BM25 Only", "bm25"), ("Hybrid RRF", "hybrid")):
        m = metrics[key]
        print(
            f"{label:<16} "
            f"{100.0 * m['r1'] / total:>11.1f}% "
            f"{100.0 * m['rk'] / total:>11.1f}% "
            f"{m['mrr'] / total:>12.3f}"
        )
    print("=" * 68)

    if misses:
        print(f"\nMissed by Hybrid RRF ({len(misses)}/{total}):")
        for line in misses:
            print(line)
    else:
        print("\n✅ All evaluation queries matched within top-K!")


if __name__ == "__main__":
    main()
