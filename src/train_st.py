"""
Contrastive Retrieval Fine-Tuning with Sentence-Transformers & ModernBERT.

Citation for ModernBERT:
  Warner, B., Chaffin, A., Clavié, B., Weller, O., Hallström, O., Taghadouini, S.,
  Gallagher, A., Biswas, R., Ladhak, F., Aarsen, T., Cooper, N., Adams, G.,
  Howard, J., & Poli, I. (2024). Smarter, Better, Faster, Longer: A Modern
  Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning
  and Inference. arXiv preprint arXiv:2412.13663. https://arxiv.org/abs/2412.13663
"""

import os
import csv

# Prevent CUDA memory fragmentation on 8GB GPUs
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerModelCardData,
    InputExample,
    losses,
)
from torch.utils.data import DataLoader
from optiDataset import clean_pdf_text

# 1. Configuration & Paths
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
csv_path = os.path.join(project_root, "synthetic_queries.csv")
feedback_csv_path = os.path.join(project_root, "feedback_log.csv")

PRETRAINED_RETRIEVAL_MODEL = "Alibaba-NLP/gte-modernbert-base"
output_model_path = os.path.join(project_root, "bert-sentence-transformer")

MODERNBERT_BIBTEX = """
@misc{warner2024smarterbetterfasterlonger,
      title={Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference},
      author={Benjamin Warner and Antoine Chaffin and Benjamin Clavié and Orion Weller and Oskar Hallström and Said Taghadouini and Alexis Gallagher and Raja Biswas and Faisal Ladhak and Tom Aarsen and Nathan Cooper and Griffin Adams and Jeremy Howard and Iacopo Poli},
      year={2024},
      eprint={2412.13663},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.13663},
}
""".strip()


def load_base_sentence_transformer() -> SentenceTransformer:
    model_card_data = SentenceTransformerModelCardData(
        language="en",
        license="apache-2.0",
        model_name="ModernBERT HVAC/BMS Technical Search Encoder",
    )
    print(f"Loading retrieval-pretrained ModernBERT model: {PRETRAINED_RETRIEVAL_MODEL}...")
    return SentenceTransformer(
        PRETRAINED_RETRIEVAL_MODEL,
        model_card_data=model_card_data,
    )


def append_modernbert_citation(readme_path: str):
    """Ensure the generated SentenceTransformer README includes the ModernBERT developer citation."""
    if not os.path.exists(readme_path):
        return
    with open(readme_path, "r", encoding="utf-8") as f:
        content = f.read()
    if "warner2024smarterbetterfasterlonger" not in content:
        citation_block = f"\n#### ModernBERT\n```bibtex\n{MODERNBERT_BIBTEX}\n```\n"
        if "## Citation" in content:
            content = content + citation_block
        else:
            content = content + f"\n## Citation\n\n### BibTeX\n{citation_block}"
        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)


def main():
    if not os.path.exists(csv_path):
        print(f"Error: Training data not found at {csv_path}")
        return

    model = load_base_sentence_transformer()
    # Ensure master weights are FP32 so PyTorch AMP GradScaler (use_amp=True) does not fail
    # when a Hugging Face checkpoint (like gte-modernbert-base) defaults to float16 weights.
    model = model.float()
    # Cap sequence length to 512 tokens for paragraph retrieval to fit comfortably in 8GB VRAM
    model.max_seq_length = 512

    # Load and clean CSV training pairs
    print("Loading training pairs from CSV...")
    train_examples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row["query"].strip()
            paragraph = clean_pdf_text(row["paragraph"].strip())
            if query and paragraph:
                train_examples.append(InputExample(texts=[query, paragraph]))

    synthetic_count = len(train_examples)
    print(f"Loaded {synthetic_count} synthetic training pairs from {csv_path}.")

    # Also incorporate real technician 👍 feedback pairs from feedback_log.csv (weighted 2x)
    if os.path.exists(feedback_csv_path):
        up_count = 0
        down_count = 0
        with open(feedback_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rating = (row.get("rating") or "").strip().lower()
                if rating == "up":
                    query = (row.get("search_query") or row.get("user_query") or "").strip()
                    paragraph = clean_pdf_text((row.get("paragraph") or "").strip())
                    if query and paragraph:
                        # Weight real field-validated queries 2x relative to synthetic pairs
                        train_examples.append(InputExample(texts=[query, paragraph]))
                        train_examples.append(InputExample(texts=[query, paragraph]))
                        up_count += 1
                elif rating == "down":
                    down_count += 1
        print(
            f"Loaded {up_count} real technician 👍 pairs (weighted 2x) from {feedback_csv_path} "
            f"({down_count} 👎 entries logged for review)."
        )

    print(f"Total training examples: {len(train_examples)}")

    # Batch size 8 with mixed precision (use_amp=True) fits in 8GB VRAM while providing 7 in-batch negatives
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=8)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    epochs = 3
    warmup_steps = max(10, int(len(train_dataloader) * epochs * 0.1))

    print("Starting Contrastive Learning training phase...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        warmup_steps=warmup_steps,
        use_amp=True,
        show_progress_bar=True,
    )

    # Save the final search-ready model and invalidate any stale embedding cache
    model.save(output_model_path)
    append_modernbert_citation(os.path.join(output_model_path, "README.md"))

    cache_file = os.path.join(output_model_path, "corpus_embeddings.pt")
    if os.path.exists(cache_file):
        os.remove(cache_file)

    print(f"Training complete! Search-ready model saved to {output_model_path}")


if __name__ == "__main__":
    main()
