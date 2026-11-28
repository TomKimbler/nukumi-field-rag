"""
Interactive Hybrid Search Engine (ModernBERT Semantic + BM25 Lexical with Reciprocal Rank Fusion)
for HVAC/BMS Manuals.

Powered by Sentence-Transformers and ModernBERT (Warner et al., 2024, arXiv:2412.13663).
"""

import hashlib
import math
import os
import re
from collections import Counter, defaultdict
import torch
from datasets import Dataset, load_from_disk
from sentence_transformers import SentenceTransformer, util
from optiDataset import (
    build_active_pdf_category_map,
    format_context_text,
    generate_corpus_paragraphs,
    iter_active_pdf_files,
)

# 1. Configuration & Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

pdf_dir = os.path.join(project_root, "pdfFiles")
corpus_path = os.path.join(project_root, "search_corpus_dataset")
model_path = os.path.join(project_root, "bert-sentence-transformer")
embeddings_cache_path = os.path.join(model_path, "corpus_embeddings.pt")

# Hybrid Search Weights for Reciprocal Rank Fusion (RRF)
SEMANTIC_WEIGHT = 0.65
BM25_WEIGHT = 0.35
RRF_K = 60

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for",
    "from", "how", "i", "in", "into", "is", "it", "its", "me", "my", "of", "on",
    "or", "should", "tell", "that", "the", "their", "there", "these", "this",
    "to", "was", "what", "when", "where", "which", "who", "why", "will", "with",
    "would", "you", "your",
}


def tokenize_for_bm25(text: str) -> list[str]:
    """
    Technical-aware tokenizer for BM25:
    - Preserves hyphenated model/part numbers & BACnet points (e.g., '31-00725', 'av-123', 'web-rl6n')
    - Also indexes constituent alphanumeric parts so both 'WEB-RL6N' and 'RL6N' match
    - Filters conversational stopwords so question phrasing doesn't dilute keyword matches
    """
    text_lower = text.lower()
    raw_tokens = re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)*", text_lower)
    tokens = []
    for tok in raw_tokens:
        if tok in STOPWORDS:
            continue
        tokens.append(tok)
        if "-" in tok:
            for part in tok.split("-"):
                if len(part) >= 2 and part not in STOPWORDS:
                    tokens.append(part)
    return tokens


class BM25Index:
    """Fast, zero-dependency inverted-index BM25Okapi implementation tuned for technical manuals."""

    def __init__(
        self,
        corpus: list[str],
        file_names: list[str] | None = None,
        categories: list[str] | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ):
        self.k1 = k1
        self.b = b
        self.corpus_size = len(corpus)
        self.doc_len = [0] * self.corpus_size
        self.inverted_index: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.idf: dict[str, float] = {}

        total_len = 0
        for doc_id, doc in enumerate(corpus):
            prefix_parts = []
            if categories and doc_id < len(categories) and categories[doc_id]:
                prefix_parts.append(categories[doc_id])
            if file_names and doc_id < len(file_names) and file_names[doc_id]:
                prefix_parts.append(os.path.splitext(file_names[doc_id])[0])
            index_text = f"{' '.join(prefix_parts)} {doc}" if prefix_parts else doc
            tokens = tokenize_for_bm25(index_text)
            length = len(tokens)
            self.doc_len[doc_id] = length
            total_len += length

            counts = Counter(tokens)
            for term, freq in counts.items():
                self.inverted_index[term].append((doc_id, freq))

        self.avgdl = (total_len / self.corpus_size) if self.corpus_size > 0 else 1.0

        for term, postings in self.inverted_index.items():
            df = len(postings)
            self.idf[term] = math.log(1.0 + (self.corpus_size - df + 0.5) / (df + 0.5))

    def search(self, query: str, top_k: int = 50) -> list[dict]:
        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return []

        scores: dict[int, float] = defaultdict(float)
        for term in query_tokens:
            if term not in self.inverted_index:
                continue
            idf = self.idf[term]
            for doc_id, freq in self.inverted_index[term]:
                dl = self.doc_len[doc_id]
                denom = freq + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                scores[doc_id] += idf * (freq * (self.k1 + 1.0)) / denom

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [{"corpus_id": doc_id, "score": score} for doc_id, score in ranked]


def hybrid_search(
    query: str,
    model: SentenceTransformer,
    paragraph_embeddings: torch.Tensor,
    bm25_index: BM25Index,
    top_k: int = 3,
    candidate_pool: int = 50,
) -> list[dict]:
    """
    Combine ModernBERT dense semantic search and BM25 lexical search using
    Reciprocal Rank Fusion (RRF).
    """
    query_embedding = model.encode(query, convert_to_tensor=True, normalize_embeddings=True)
    semantic_hits = util.semantic_search(
        query_embedding, paragraph_embeddings, top_k=candidate_pool
    )[0]
    bm25_hits = bm25_index.search(query, top_k=candidate_pool)

    rrf_scores: dict[int, float] = defaultdict(float)
    semantic_info: dict[int, tuple[int, float]] = {}
    bm25_info: dict[int, tuple[int, float]] = {}

    for rank, hit in enumerate(semantic_hits, start=1):
        doc_id = int(hit["corpus_id"])
        score = float(hit["score"])
        semantic_info[doc_id] = (rank, score)
        rrf_scores[doc_id] += SEMANTIC_WEIGHT / (RRF_K + rank)

    for rank, hit in enumerate(bm25_hits, start=1):
        doc_id = int(hit["corpus_id"])
        score = float(hit["score"])
        bm25_info[doc_id] = (rank, score)
        rrf_scores[doc_id] += BM25_WEIGHT / (RRF_K + rank)

    # Normalize RRF score to a 0..100% relative scale based on the theoretical maximum (rank 1 in both)
    max_possible_rrf = (SEMANTIC_WEIGHT + BM25_WEIGHT) / (RRF_K + 1)

    top_doc_ids = sorted(rrf_scores.keys(), key=lambda d: rrf_scores[d], reverse=True)[:top_k]
    results = []
    for doc_id in top_doc_ids:
        sem_rank, sem_score = semantic_info.get(doc_id, (None, 0.0))
        lex_rank, lex_score = bm25_info.get(doc_id, (None, 0.0))
        results.append({
            "corpus_id": doc_id,
            "hybrid_score": rrf_scores[doc_id] / max_possible_rrf,
            "semantic_score": sem_score,
            "semantic_rank": sem_rank,
            "bm25_score": lex_score,
            "bm25_rank": lex_rank,
        })
    return results


def load_or_build_corpus() -> Dataset | None:
    """
    Load the clean paragraph corpus (with file_name, page, paragraph) from disk,
    or build it automatically from pdfFiles if it does not exist yet.
    """
    if os.path.exists(corpus_path):
        print(f"Loading search corpus from {corpus_path}...")
        return load_from_disk(corpus_path)

    if not os.path.exists(pdf_dir):
        print(f"Error: Neither {corpus_path} nor {pdf_dir} exists.")
        return None

    print(f"Building clean paragraph search corpus from {pdf_dir} (first-time setup)...")
    corpus_dataset = Dataset.from_generator(
        generate_corpus_paragraphs,
        gen_kwargs={"pdf_directory": pdf_dir},
    )
    corpus_dataset.save_to_disk(corpus_path)
    print(f"Saved {len(corpus_dataset)} clean paragraphs to {corpus_path}")
    return corpus_dataset


EMBEDDING_CACHE_VERSION = 2


def resolve_corpus_categories(
    file_names: list[str],
    dataset_categories: list[str] | None = None,
) -> list[str]:
    """
    Resolve each chunk's category breadcrumb, preferring the live `pdfFiles/` subfolder
    structure so reorganizing PDFs into folders takes effect immediately on startup.
    """
    live_map = build_active_pdf_category_map(pdf_dir) if os.path.exists(pdf_dir) else {}
    resolved: list[str] = []
    for i, fname in enumerate(file_names):
        if fname in live_map:
            resolved.append(live_map[fname])
        elif dataset_categories and i < len(dataset_categories):
            resolved.append(dataset_categories[i] or "")
        else:
            resolved.append("")
    return resolved


def _compute_metadata_signature(file_names: list[str], categories: list[str]) -> str:
    h = hashlib.sha256()
    for fn, cat in zip(file_names, categories):
        h.update(f"{cat}|{fn}\n".encode("utf-8"))
    return h.hexdigest()[:16]


def load_or_compute_embeddings(
    model: SentenceTransformer,
    paragraphs: list[str],
    file_names: list[str] | None = None,
    categories: list[str] | None = None,
) -> torch.Tensor:
    """
    Load cached paragraph embeddings if they match the current corpus size and folder/manual
    metadata, otherwise encode all context-enriched paragraphs and cache them to disk.
    """
    device = model.device
    fnames = file_names or [""] * len(paragraphs)
    cats = categories or [""] * len(paragraphs)
    meta_sig = _compute_metadata_signature(fnames, cats)

    if os.path.exists(embeddings_cache_path):
        try:
            cached = torch.load(embeddings_cache_path, map_location=device, weights_only=True)
            if (
                isinstance(cached, dict)
                and cached.get("cache_version") == EMBEDDING_CACHE_VERSION
                and cached.get("meta_sig") == meta_sig
                and cached.get("num_paragraphs") == len(paragraphs)
                and cached["embeddings"].shape[0] == len(paragraphs)
            ):
                print(f"Loaded cached embeddings for {len(paragraphs)} paragraphs from {embeddings_cache_path}")
                return cached["embeddings"]
        except Exception as e:
            print(f"Recomputing embeddings (cache could not be loaded: {e})")

    texts_to_encode = [
        format_context_text(p, fn, cat) for p, fn, cat in zip(paragraphs, fnames, cats)
    ]
    print(f"Indexing {len(paragraphs)} context-enriched paragraphs into vector space...")
    embeddings = model.encode(
        texts_to_encode,
        batch_size=32,
        convert_to_tensor=True,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    torch.save(
        {
            "cache_version": EMBEDDING_CACHE_VERSION,
            "meta_sig": meta_sig,
            "num_paragraphs": len(paragraphs),
            "embeddings": embeddings.cpu(),
        },
        embeddings_cache_path,
    )
    print(f"Cached embeddings to {embeddings_cache_path} for fast future startups.")
    return embeddings


def filter_active_corpus(
    paragraphs: list[str],
    file_names: list[str],
    pages: list[int],
    paragraph_embeddings: torch.Tensor,
    categories: list[str] | None = None,
) -> tuple[list[str], list[str], list[int], torch.Tensor, list[str]]:
    """
    Live subfolder exclusion filter: if `pdfFiles/` exists, check which PDFs are currently
    active (not inside an `ignore_` or `exclude_` folder or prefixed with `ignore_`/`exclude_`).
    Any excluded PDFs are masked out in memory without invalidating `corpus_embeddings.pt`.
    """
    cats = categories if categories is not None else resolve_corpus_categories(file_names)
    if not os.path.exists(pdf_dir):
        return paragraphs, file_names, pages, paragraph_embeddings, cats

    active_pdfs = iter_active_pdf_files(pdf_dir)
    if not active_pdfs:
        return paragraphs, file_names, pages, paragraph_embeddings, cats

    active_names = {fname for fname, _ in active_pdfs}
    keep_indices = [i for i, fname in enumerate(file_names) if fname in active_names]

    if len(keep_indices) == len(paragraphs) or len(keep_indices) == 0:
        return paragraphs, file_names, pages, paragraph_embeddings, cats

    excluded_paragraphs = len(paragraphs) - len(keep_indices)
    excluded_files = len(set(file_names) - active_names)
    print(
        f"Live filter: excluded {excluded_files} manual(s) ({excluded_paragraphs} paragraphs) "
        f"marked with ignore_/exclude_."
    )
    filtered_paragraphs = [paragraphs[i] for i in keep_indices]
    filtered_files = [file_names[i] for i in keep_indices]
    filtered_pages = [pages[i] for i in keep_indices]
    filtered_cats = [cats[i] for i in keep_indices]
    idx_tensor = torch.tensor(keep_indices, dtype=torch.long, device=paragraph_embeddings.device)
    filtered_embeddings = paragraph_embeddings.index_select(0, idx_tensor)
    return filtered_paragraphs, filtered_files, filtered_pages, filtered_embeddings, filtered_cats


def main():
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Did you run train_st.py?")
        return

    corpus_dataset = load_or_build_corpus()
    if corpus_dataset is None or len(corpus_dataset) == 0:
        print("Error: Search corpus is empty.")
        return

    print(f"Loading search model from {model_path}...")
    model = SentenceTransformer(model_path)

    paragraphs = list(corpus_dataset["paragraph"])
    file_names = list(corpus_dataset["file_name"])
    pages = list(corpus_dataset["page"])
    raw_cats = list(corpus_dataset["category"]) if "category" in corpus_dataset.column_names else None
    categories = resolve_corpus_categories(file_names, raw_cats)

    paragraph_embeddings = load_or_compute_embeddings(
        model, paragraphs, file_names=file_names, categories=categories
    )
    paragraphs, file_names, pages, paragraph_embeddings, categories = filter_active_corpus(
        paragraphs, file_names, pages, paragraph_embeddings, categories=categories
    )

    print(f"Building BM25 lexical index for {len(paragraphs)} paragraphs...")
    bm25_index = BM25Index(paragraphs, file_names=file_names, categories=categories)

    print("\n" + "=" * 60)
    print("🚀 HYBRID SEARCH ENGINE READY (ModernBERT + BM25 RRF)")
    print("=" * 60)
    print("Type a question or part number to search your manuals. Type 'quit' or 'exit' to stop.")

    # Interactive Search Loop
    while True:
        try:
            query = input("\n🔍 Search: ")
        except (EOFError, KeyboardInterrupt):
            print("\nShutting down search engine. Goodbye!")
            break

        if query.lower() in ["quit", "exit", "q"]:
            print("Shutting down search engine. Goodbye!")
            break

        if not query.strip():
            continue

        hits = hybrid_search(
            query=query,
            model=model,
            paragraph_embeddings=paragraph_embeddings,
            bm25_index=bm25_index,
            top_k=3,
        )

        print("\n--- Top Hybrid Results ---")
        for i, hit in enumerate(hits):
            para_id = hit["corpus_id"]
            matched_paragraph = paragraphs[para_id]
            source_file = file_names[para_id]
            source_page = pages[para_id]
            source_cat = categories[para_id]

            sem_str = (
                f"{hit['semantic_score']:.2f} (#{hit['semantic_rank']})"
                if hit["semantic_rank"]
                else "n/a"
            )
            bm25_str = (
                f"{hit['bm25_score']:.1f} (#{hit['bm25_rank']})"
                if hit["bm25_rank"]
                else "n/a"
            )

            print(
                f"\nResult #{i + 1} | Hybrid: {hit['hybrid_score']:.2f} "
                f"[Semantic: {sem_str} | BM25: {bm25_str}]"
            )
            cat_prefix = f"[{source_cat}] " if source_cat else ""
            print(f"📄 Source: {cat_prefix}{source_file} (Page {source_page})")
            print(f"{matched_paragraph}")
            print("-" * 60)


if __name__ == "__main__":
    main()
