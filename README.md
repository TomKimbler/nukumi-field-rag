# Nukumi — Technical Manual Hybrid Search & Field Assistant

A domain-adapted **Hybrid Search (ModernBERT + BM25)** and conversational **RAG Mobile PWA** designed for field technicians working with technical PDF manuals on a modest 8 GB GPU (configured out-of-the-box for HVAC & Building Management Systems, and easily adaptable to any technical trade via [`config.json`](config.json)).

---

## Features

- **Hybrid Retrieval (ModernBERT + BM25 Reciprocal Rank Fusion)**: Combines dense semantic search (`Alibaba-NLP/gte-modernbert-base`, fine-tuned on your manuals) with a zero-dependency inverted-index `BM25Okapi` lexical search that matches exact part numbers, terminal IDs (`BO-1`, `AV-123`), and manual titles.
- **Self-Improving Feedback Loop (`👍 / 👎 / 🎯 Best Match`)**: Technicians can rate answers or mark specific manual excerpts as the `🎯 Best Match` directly on their phone; `train_st.py` automatically folds `👍` pairs from `feedback_log.csv` into the next training run at 2× weight.
- **Mobile & Tablet PWA with Hot-Swappable HTML Themes (`themes/*.html`)**: Zero-dependency Python HTTP server (`127.0.0.1:8000`) with a slide-out Hamburger Menu (`☰`), 6-digit PIN gate (`.access_pin`), model lock pill (`🔒`), collapsible verbatim citations drawer, and clickable `📄 Source: Manual.pdf (Page X)` links that open the PDF right to Page `X`. Includes 3 built-in field themes (**Slate Dark**, **Daylight Rooftop** for direct sunlight, and **Amber Industrial** for dark mechanical rooms) and auto-discovers any custom `.html` file dropped into `themes/`.
- **Multi-Turn Field Context**: Automatically carries equipment context across follow-up questions (*"How do I factory reset it?"*) and enforces explicit device attribution so model-specific procedures are never mistaken for universal rules.
- **Universal Engineering Math & Physics Formulas**: Keeps proprietary equipment specs (terminals, BACnet IDs, DIP switches) strictly locked to your PDF manuals while seamlessly solving field calculations (Ohm's/Watt's Law, electric heater BTU/hr output, airflow/hydronic heat transfer, gas laws, 4–20mA / 0–10VDC signal scaling) with step-by-step formulas and a distinct `🧮 Source: Standard Engineering Formula (Not from PDF Manual)` badge.
- **Domain-Aware Subfolder Taxonomy & Instant `ignore_` / `exclude_` Filtering**: Organize `pdfFiles/` into any trade/vendor/equipment subfolder hierarchy (e.g., `pdfFiles/Honeywell/Optimizer_Advanced/`, `pdfFiles/Electrical/Switchgear/`). Subfolder names are automatically injected as category context into BM25, ModernBERT embeddings, synthetic query generation, RAG prompts, and `lock` filters. Prefixing any subfolder or PDF with `ignore_` or `exclude_` masks those manuals out of live search immediately on startup.

---

## Hardware & Software Prerequisites

- **OS**: Linux, macOS, or Windows (WSL2 recommended on Windows)
- **Python**: Python 3.10+
- **GPU**: NVIDIA GPU with **8 GB+ VRAM** recommended for training & local LLM inference (can also run search on CPU via `CPU_RETRIEVER=1`)
- **Ollama**: Installed and running locally ([https://ollama.com](https://ollama.com))

---

## Step-by-Step Setup (From Scratch)

### 1. Clone, Create Virtual Environment & Install Dependencies
```bash
git clone https://github.com/<your-username>/<your-repo-name>.git
cd <your-repo-name>

# Create a Python virtual environment in .venv
python3 -m venv .venv

# Activate the virtual environment (run this whenever you open a new terminal)
source .venv/bin/activate

# Install required packages
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Pull the Local Ollama Model
By default, [`Modelfile`](Modelfile) and [`src/generate_queries.py`](src/generate_queries.py) use `gemma4:e2b` (you can change this in `Modelfile` and `src/generate_queries.py` to any local Ollama model):
```bash
ollama pull gemma4:e2b
```

### 3. Add Your PDF Manuals & Configure Your Domain
1. Place your `.pdf` technical manuals inside **`pdfFiles/`** organized into descriptive subfolders (e.g., `pdfFiles/Honeywell/Optimizer_Advanced/`, `pdfFiles/Electrical/Switchgear/`). Subfolder names automatically feed trade/vendor context into retrieval, query generation, and `lock` filtering; prefix any subfolder or file with `ignore_` or `exclude_` to temporarily exclude it from search without moving files.
2. *(Optional)* If adapting for an industry other than HVAC/BMS (e.g., industrial PLC/automation, automotive/diesel service, medical devices, aviation, IT runbooks):
   - Edit **[`config.json`](config.json)** to set your `app_name`, `domain_role`, `domain_equipment_examples`, and sample questions.
   - Update the `SYSTEM` persona line in **[`Modelfile`](Modelfile)** to match your trade.

### 4. Build the Search Corpus & Train the Domain Retriever
With your virtual environment active (`source .venv/bin/activate`), run the 3-step data & training pipeline once:

```bash
# Step 4a: Extract and clean paragraphs (with file_name & page metadata) from pdfFiles/
python src/optiDataset.py

# Step 4b: Generate synthetic domain questions paired with manual paragraphs (~10-15 mins)
python src/generate_queries.py

# IMPORTANT on 8 GB GPUs: Free Ollama's VRAM before starting PyTorch fine-tuning
ollama stop gemma4:e2b

# Step 4c: Fine-tune the ModernBERT SentenceTransformer on synthetic_queries.csv (+ any 👍 feedback)
python src/train_st.py
```

---

## Running the Assistant

*(Make sure your virtual environment is active in your terminal: `source .venv/bin/activate`)*

### Option A: Mobile PWA Web Server (Tailscale Serve / Funnel)
```bash
# Start the local web server (binds to 127.0.0.1:8000 and prints your 6-digit Access PIN)
python src/web_app.py
```
In a second terminal, expose port `8000` over HTTPS using [Tailscale](https://tailscale.com):
```bash
# Private to devices on your Tailnet (foreground; closes when you exit the terminal):
tailscale serve 8000

# Or public HTTPS link for field testers WITHOUT the Tailscale app (protected by your Access PIN):
tailscale funnel 8000
```
To run Tailscale in the background without leaving a terminal open (persists across reboots via `tailscaled`), add `--bg`:
```bash
# Background persistent Serve (Tailnet-only) or Funnel (public HTTPS):
tailscale serve --bg 8000
tailscale funnel --bg 8000

# Check status or stop the background Funnel anytime:
tailscale funnel status
tailscale funnel --bg off
```
*(If Tailscale prompts for elevated privileges, prefix with `sudo` or grant your user operator permissions once via `sudo tailscale set --operator=$USER`.)*

Open the `https://<machine>.<tailnet>.ts.net` URL on your phone, enter the 6-digit PIN from `.access_pin` (editable anytime), and tap **"Add to Home Screen"**.

#### Optional: Run as an Auto-Restarting Background Service (Linux `systemd`)
If you are testing or experimenting, simply run `python src/web_app.py` in a terminal as shown above. When you are ready for unattended field use (auto-restart on crash and auto-start on boot), run the optional service installer:
```bash
# Install and start the user-level systemd background service
./install_service.sh

# Useful service commands:
systemctl --user status nukumi      # Check status
journalctl --user -u nukumi -f      # View live logs & Access PIN
systemctl --user stop nukumi        # Stop service (e.g., before running train_st.py)
./install_service.sh --uninstall    # Remove background service and return to manual mode
```
*(External Tailscale exposure is manual by default. To automatically enable background Tailscale Funnel during install, uncomment the `tailscale funnel --bg 8000` line in `install_service.sh`.)*

### Option B: Terminal CLI Conversational Assistant
```bash
python src/assistant.py
```
*Supported in-chat commands (no `/` prefix required)*:
- `guide` or `help` — Show available commands
- `citations` or `sources` — View the 4 verbatim manual excerpts & Hybrid/Semantic/BM25 scores for the last answer
- `good` or `bad [note]` — Log `👍` or `👎` feedback to `feedback_log.csv` for future `train_st.py` runs
- `lock <model>` / `unlock` — Lock retrieval to a specific controller/manual (e.g., `lock CIPer 30`) or unlock
- `reset` or `new` — Clear conversation history when moving to a new piece of equipment
- `quit` or `exit` — Exit

*(Tip: If you want 100% of GPU VRAM reserved for a larger Ollama model, run with `CPU_RETRIEVER=1 python src/web_app.py` to run ModernBERT on CPU in ~20 ms/query.)*

### Option C: Standalone Hybrid Search (No LLM Generation)
```bash
python src/test_search.py
```

---

## Continuous Improvement Loop (`feedback_log.csv`)

As technicians use the Mobile PWA or CLI in the field:
1. Tapping **`👍 Helpful`** on an answer—or opening the **`📚 View 4 Verbatim Manual Excerpts`** drawer and tapping **`🎯 Best Match`** on a specific excerpt—saves the contextualized query and paragraph to `feedback_log.csv`.
2. Tapping **`👎 Needs Work`** records the query and an optional technician note so you can see where manuals or retrieval need attention.
3. Re-running `.venv/bin/python src/train_st.py` automatically merges all `👍` pairs from `feedback_log.csv` at **2× weight** alongside `synthetic_queries.csv` and invalidates the old embedding cache (`corpus_embeddings.pt`).

---

## Project Structure

| File | Purpose |
| :--- | :--- |
| `config.json` | Domain configuration (`app_name`, `domain_role`, equipment examples, sample prompts) |
| `Modelfile` | Ollama base model, temperature, context window, and safety-first system prompt (hot-reloaded on every query) |
| `src/optiDataset.py` | Extracts & cleans PDFs from `pdfFiles/` into `search_corpus_dataset/` |
| `src/generate_queries.py` | Generates synthetic technician training questions (`synthetic_queries.csv`) via Ollama |
| `src/train_st.py` | Fine-tunes `Alibaba-NLP/gte-modernbert-base` on `synthetic_queries.csv` + `feedback_log.csv` |
| `src/test_search.py` | Standalone Hybrid Search CLI (`BM25Index` + ModernBERT RRF + `corpus_embeddings.pt` caching) |
| `src/assistant.py` | Multi-turn conversational RAG CLI assistant |
| `src/web_app.py` | Zero-dependency Mobile & Tablet PWA server with PIN auth, streaming SSE, `#page=X` PDF viewer, and live theme loader |
| `themes/*.html` | Hot-swappable self-contained HTML themes (`slate-dark.html`, `daylight-rooftop.html`, `amber-industrial.html`) + shared `themes/app.js` |
| `install_service.sh` | Optional Linux `systemd --user` background service installer (`--uninstall` to remove) |

---

## Citations & Attribution

This project builds upon **ModernBERT** (developed by Answer.AI, LightOn, and contributors under the Apache 2.0 License) and **Sentence-Transformers**.

### ModernBERT
```bibtex
@misc{warner2024smarterbetterfasterlonger,
      title={Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder for Fast, Memory Efficient, and Long Context Finetuning and Inference},
      author={Benjamin Warner and Antoine Chaffin and Benjamin Clavié and Orion Weller and Oskar Hallström and Said Taghadouini and Alexis Gallagher and Raja Biswas and Faisal Ladhak and Tom Aarsen and Nathan Cooper and Griffin Adams and Jeremy Howard and Iacopo Poli},
      year={2024},
      eprint={2412.13663},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2412.13663},
}
```

### Sentence-BERT
```bibtex
@inproceedings{reimers-2019-sentence-bert,
    title = "Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks",
    author = "Reimers, Nils and Gurevych, Iryna",
    booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing",
    month = "11",
    year = "2019",
    publisher = "Association for Computational Linguistics",
    url = "https://arxiv.org/abs/1908.10084",
}
```

---

## License

This project is licensed under the **MIT License** — see [`LICENSE`](LICENSE) for details. Third-party models and libraries (such as ModernBERT and Sentence-Transformers) remain subject to their respective licenses.
