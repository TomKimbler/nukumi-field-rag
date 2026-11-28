"""
Field Technician Conversational RAG Assistant (Mobile / Chat-Ready).

Combines Hybrid Search (ModernBERT + BM25 Reciprocal Rank Fusion) with a local
Ollama LLM (configured via Modelfile) for multi-turn field troubleshooting.

Citation for ModernBERT:
  Warner et al. (2024), arXiv:2412.13663 — https://arxiv.org/abs/2412.13663
"""

import csv
from datetime import datetime, timezone
import os
import re
import threading
import ollama
import torch
from sentence_transformers import SentenceTransformer
from optiDataset import iter_active_pdf_files
from test_search import (
    BM25Index,
    filter_active_corpus,
    hybrid_search,
    load_or_build_corpus,
    load_or_compute_embeddings,
    model_path,
    pdf_dir,
    resolve_corpus_categories,
)

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
modelfile_path = os.path.join(project_root, "Modelfile")
feedback_csv_path = os.path.join(project_root, "feedback_log.csv")
_feedback_lock = threading.Lock()

FEEDBACK_FIELDNAMES = [
    "timestamp",
    "rating",
    "user_query",
    "search_query",
    "active_device",
    "file_name",
    "page",
    "paragraph",
    "note",
]


def log_feedback(
    rating: str,
    user_query: str,
    search_query: str,
    active_device: str | None,
    file_name: str,
    page: int | str,
    paragraph: str,
    note: str = "",
):
    """Thread-safely append a technician feedback record ('up' or 'down') to feedback_log.csv."""
    with _feedback_lock:
        file_exists = os.path.exists(feedback_csv_path) and os.path.getsize(feedback_csv_path) > 0
        with open(feedback_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FEEDBACK_FIELDNAMES)
            if not file_exists:
                writer.writeheader()
            writer.writerow({
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "rating": rating,
                "user_query": user_query,
                "search_query": search_query,
                "active_device": active_device or "",
                "file_name": file_name,
                "page": page,
                "paragraph": paragraph,
                "note": note,
            })

DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e2b")

FOLLOWUP_PRONOUNS = {
    "it", "its", "this", "that", "these", "those", "they", "them", "same",
}


def load_modelfile_config(path: str) -> tuple[str, float, int, str]:
    """
    Parse model name, temperature, num_ctx, and SYSTEM prompt directly from Modelfile
    so changes to Modelfile take effect immediately without requiring `ollama create`.
    """
    base_model = DEFAULT_OLLAMA_MODEL
    temperature = 0.2
    num_ctx = 8192
    system_prompt = (
        "You are a technical support assistant for HVAC/BMS technicians. "
        "Answer strictly from the provided manual context and cite the manual filename and page."
    )

    if not os.path.exists(path):
        return base_model, temperature, num_ctx, system_prompt

    with open(path, "r", encoding="utf-8") as f:
        content = f.read()

    from_match = re.search(r"^FROM\s+(\S+)", content, re.MULTILINE)
    if from_match:
        base_model = os.environ.get("OLLAMA_MODEL", from_match.group(1).strip())

    temp_match = re.search(r"^PARAMETER\s+temperature\s+([0-9.]+)", content, re.MULTILINE)
    if temp_match:
        temperature = float(temp_match.group(1))

    ctx_match = re.search(r"^PARAMETER\s+num_ctx\s+(\d+)", content, re.MULTILINE)
    if ctx_match:
        num_ctx = int(ctx_match.group(1))

    sys_match = re.search(r'SYSTEM\s+"""(.*?)"""', content, re.DOTALL)
    if sys_match:
        system_prompt = sys_match.group(1).strip()

    return base_model, temperature, num_ctx, system_prompt


def build_contextual_search_query(
    user_message: str,
    active_device: str | None,
    last_user_queries: list[str],
) -> str:
    """
    Resolve pronouns and multi-turn context for retrieval:
    If the technician asks a follow-up like 'How do I factory reset it?' or has
    an active device pinned, include that context in the search query so ModernBERT
    and BM25 retrieve the right manual.
    """
    words = set(re.findall(r"[a-z0-9-]+", user_message.lower()))
    search_query = user_message.strip()

    if active_device and active_device.lower() not in search_query.lower():
        search_query = f"{active_device}: {search_query}"
    elif last_user_queries and (len(words) <= 7 or (words & FOLLOWUP_PRONOUNS)):
        # Carry forward key technical tokens from the previous user turn
        prev_context = last_user_queries[-1]
        search_query = f"{prev_context} — {search_query}"

    return search_query


def retrieve_context_hits(
    search_query: str,
    model: SentenceTransformer,
    paragraph_embeddings: torch.Tensor,
    bm25_index: BM25Index,
    file_names: list[str],
    active_device: str | None = None,
    top_k: int = 4,
    categories: list[str] | None = None,
) -> list[dict]:
    """
    Run Hybrid Search (ModernBERT + BM25 RRF) and optionally prioritize/filter
    manuals matching `active_device` (by subfolder category or filename).
    """
    pool_size = 80 if active_device else 50
    hits = hybrid_search(
        query=search_query,
        model=model,
        paragraph_embeddings=paragraph_embeddings,
        bm25_index=bm25_index,
        top_k=pool_size,
        candidate_pool=pool_size,
    )

    if active_device:
        device_terms = [t.lower() for t in active_device.split() if t.strip()]
        filtered = []
        for h in hits:
            cid = h["corpus_id"]
            cat = categories[cid] if (categories and cid < len(categories)) else ""
            haystack = f"{cat} {file_names[cid]}".lower()
            if all(term in haystack for term in device_terms):
                filtered.append(h)
        if filtered:
            return filtered[:top_k]

    return hits[:top_k]


def format_rag_prompt(
    user_message: str,
    hits: list[dict],
    paragraphs: list[str],
    file_names: list[str],
    pages: list[int],
    active_device: str | None,
    categories: list[str] | None = None,
) -> str:
    """Build the grounded RAG prompt combining retrieved manual excerpts and the technician's question."""
    excerpts = []
    for idx, hit in enumerate(hits, start=1):
        cid = hit["corpus_id"]
        src_file = file_names[cid]
        src_page = pages[cid]
        para_text = paragraphs[cid]
        src_cat = categories[cid] if (categories and cid < len(categories)) else ""
        cat_part = f"Category: {src_cat} | " if src_cat else ""
        excerpts.append(
            f"[Excerpt {idx} | {cat_part}Manual: {src_file} | Page: {src_page}]\n{para_text}"
        )

    context_block = "\n\n".join(excerpts)
    device_note = f"\nActive Equipment Filter: {active_device}\n" if active_device else ""

    return (
        f"Manual Context:\n{context_block}\n"
        f"{device_note}\n"
        f"Technician Question: {user_message}"
    )


def get_manual_count(file_names: list[str]) -> int:
    """Count active PDF manuals in ./pdfFiles (including subfolders, excluding ignore_/exclude_)."""
    active_pdfs = iter_active_pdf_files(pdf_dir)
    if active_pdfs:
        return len(active_pdfs)
    return len(set(file_names))


def print_guide(active_device: str | None, manual_count: int):
    device_label = active_device if active_device else f"None (searching all {manual_count} manuals)"
    print("\n" + "-" * 60)
    print("📱 FIELD ASSISTANT COMMANDS (No '/' prefix needed)")
    print("-" * 60)
    print("  guide / help        : Show this command reference")
    print("  citations / sources : View verbatim manual excerpts & scores for the last reply")
    print("  good / bad [note]   : Log 👍 or 👎 feedback on the last answer for model training")
    print("  lock <model>        : Lock search to a controller/manual (e.g. 'lock CIPer 30')")
    print("  unlock              : Clear model lock and search all manuals")
    print("  reset / new         : Clear chat history when starting on a new unit")
    print("  quit / exit         : Close the assistant")
    print(f"  Current Model Lock  : {device_label}")
    print("-" * 60)


def print_last_citations(
    last_hits: list[dict],
    paragraphs: list[str],
    file_names: list[str],
    pages: list[int],
):
    if not last_hits:
        print("\nNo citations yet—ask a technical question first!")
        return

    print("\n--- Verbatim Manual Excerpts (Last Query) ---")
    for i, hit in enumerate(last_hits, start=1):
        cid = hit["corpus_id"]
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
            f"\n[{i}] 📄 {file_names[cid]} (Page {pages[cid]}) | "
            f"Hybrid: {hit['hybrid_score']:.2f} [Semantic: {sem_str} | BM25: {bm25_str}]"
        )
        print(paragraphs[cid])
        print("-" * 50)


def main():
    if not os.path.exists(model_path):
        print(f"Error: Search model not found at {model_path}. Run train_st.py first.")
        return

    corpus_dataset = load_or_build_corpus()
    if corpus_dataset is None or len(corpus_dataset) == 0:
        print("Error: Search corpus is empty.")
        return

    # Optional: set CPU_RETRIEVER=1 to run ModernBERT on CPU (~20ms/query) and leave 100% of GPU VRAM for Ollama
    retriever_device = "cpu" if os.environ.get("CPU_RETRIEVER", "0") == "1" else None
    print(f"Loading ModernBERT search retriever from {model_path}...")
    retriever = SentenceTransformer(model_path, device=retriever_device)

    paragraphs = list(corpus_dataset["paragraph"])
    file_names = list(corpus_dataset["file_name"])
    pages = list(corpus_dataset["page"])
    raw_cats = list(corpus_dataset["category"]) if "category" in corpus_dataset.column_names else None
    categories = resolve_corpus_categories(file_names, raw_cats)
    manual_count = get_manual_count(file_names)

    paragraph_embeddings = load_or_compute_embeddings(
        retriever, paragraphs, file_names=file_names, categories=categories
    )
    if retriever.device.type == "cpu":
        paragraph_embeddings = paragraph_embeddings.cpu()
    paragraphs, file_names, pages, paragraph_embeddings, categories = filter_active_corpus(
        paragraphs, file_names, pages, paragraph_embeddings, categories=categories
    )
    manual_count = get_manual_count(file_names)

    print(f"Building BM25 lexical index over {len(paragraphs)} paragraphs ({manual_count} manuals)...")
    bm25_index = BM25Index(paragraphs, file_names=file_names, categories=categories)

    ollama_model, temperature, num_ctx, system_prompt = load_modelfile_config(modelfile_path)

    print("\n" + "=" * 60)
    print(f"🛠️  FIELD SUPPORT ASSISTANT READY (LLM: {ollama_model})")
    print("=" * 60)
    print("Ask a question, or type 'guide' for commands ('citations', 'lock <model>', 'unlock', 'reset', 'quit').")

    chat_history: list[dict[str, str]] = []
    last_user_queries: list[str] = []
    last_search_query: str = ""
    last_hits: list[dict] = []
    last_reply_is_formula: bool = False
    active_device: str | None = None

    while True:
        prompt_prefix = f"\n[🔒 {active_device}] " if active_device else "\n"
        try:
            user_input = input(f"{prompt_prefix}👷 Tech: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nShutting down assistant. Stay safe out there!")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in {"quit", "exit", "q"}:
            print("Shutting down assistant. Stay safe out there!")
            break

        if cmd in {"guide", "help", "/help", "?"}:
            print_guide(active_device, manual_count)
            continue

        if cmd in {"citations", "citation", "sources", "source"}:
            print_last_citations(last_hits, paragraphs, file_names, pages)
            continue

        if cmd in {"good", "up", "👍"}:
            if not last_hits or not last_user_queries:
                print("Ask a question first before logging feedback!")
                continue
            if last_reply_is_formula:
                log_feedback(
                    rating="up_formula",
                    user_query=last_user_queries[-1],
                    search_query=last_search_query or last_user_queries[-1],
                    active_device=active_device,
                    file_name="Standard Engineering Formula",
                    page="",
                    paragraph="",
                )
                print("👍 Formula feedback logged (excluded from manual-retrieval training).")
                continue
            top_cid = last_hits[0]["corpus_id"]
            log_feedback(
                rating="up",
                user_query=last_user_queries[-1],
                search_query=last_search_query or last_user_queries[-1],
                active_device=active_device,
                file_name=file_names[top_cid],
                page=pages[top_cid],
                paragraph=paragraphs[top_cid],
            )
            print("👍 Feedback saved to feedback_log.csv for future SentenceTransformer training!")
            continue

        if cmd in {"bad", "down", "👎"} or cmd.startswith("bad "):
            if not last_hits or not last_user_queries:
                print("Ask a question first before logging feedback!")
                continue
            note = user_input[3:].strip() if cmd.startswith("bad") else ""
            top_cid = last_hits[0]["corpus_id"]
            log_feedback(
                rating="down",
                user_query=last_user_queries[-1],
                search_query=last_search_query or last_user_queries[-1],
                active_device=active_device,
                file_name=file_names[top_cid],
                page=pages[top_cid],
                paragraph=paragraphs[top_cid],
                note=note,
            )
            print("👎 Feedback logged to feedback_log.csv for review.")
            continue

        if cmd in {"reset", "new", "clear"}:
            chat_history.clear()
            last_user_queries.clear()
            last_hits.clear()
            active_device = None
            print("🔄 Conversation history and model lock cleared.")
            continue

        if cmd in {"unlock", "unpin"}:
            active_device = None
            print(f"🔓 Model lock cleared (searching all {manual_count} manuals).")
            continue

        if (cmd == "lock" or cmd.startswith("lock ")) and not user_input.endswith("?"):
            arg = user_input[4:].strip()
            if not arg or arg.lower() in {"off", "clear", "none", "all"}:
                active_device = None
                print(f"🔓 Model lock cleared (searching all {manual_count} manuals).")
            elif len(arg.split()) <= 5:
                active_device = arg
                print(f"🔒 Locked manual filter to: '{active_device}' (type 'unlock' to clear).")
            else:
                # Longer sentence starting with 'lock'—treat as a normal question
                pass
            if len(arg.split()) <= 5:
                continue

        # 1. Contextualize follow-up query & run Hybrid Search
        search_query = build_contextual_search_query(user_input, active_device, last_user_queries)
        last_search_query = search_query
        last_hits = retrieve_context_hits(
            search_query=search_query,
            model=retriever,
            paragraph_embeddings=paragraph_embeddings,
            bm25_index=bm25_index,
            file_names=file_names,
            active_device=active_device,
            top_k=4,
            categories=categories,
        )

        # 2. Build grounded RAG prompt
        rag_user_prompt = format_rag_prompt(
            user_message=user_input,
            hits=last_hits,
            paragraphs=paragraphs,
            file_names=file_names,
            pages=pages,
            active_device=active_device,
            categories=categories,
        )

        # Re-read Modelfile dynamically so live prompt/parameter edits take effect without restarting
        ollama_model, temperature, num_ctx, system_prompt = load_modelfile_config(modelfile_path)

        # Keep recent conversation history (last 4 messages = 2 turns) so multi-turn context stays sharp
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(chat_history[-4:])
        messages.append({"role": "user", "content": rag_user_prompt})

        print("\n🤖 Assistant: ", end="", flush=True)
        assistant_reply_parts = []
        try:
            stream = ollama.chat(
                model=ollama_model,
                messages=messages,
                stream=True,
                options={
                    "temperature": temperature,
                    "num_ctx": num_ctx,
                },
            )
            for chunk in stream:
                token_text = chunk.get("message", {}).get("content", "")
                if token_text:
                    print(token_text, end="", flush=True)
                    assistant_reply_parts.append(token_text)
            print()
        except Exception as e:
            print(f"\n[Error communicating with Ollama ({ollama_model}): {e}]")
            print("Tip: Make sure Ollama is running (`ollama serve`) and showing top manual hits instead:")
            print_last_citations(last_hits, paragraphs, file_names, pages)
            continue

        full_reply = "".join(assistant_reply_parts).strip()
        if full_reply:
            # Store compact user question + assistant answer in multi-turn memory
            last_reply_is_formula = "Standard Engineering Formula" in full_reply
            chat_history.append({"role": "user", "content": user_input})
            chat_history.append({"role": "assistant", "content": full_reply})
            last_user_queries.append(user_input)


if __name__ == "__main__":
    main()
