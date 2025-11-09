
# DictaBERT Tiny Joint – Starter Project (VS Code)

This is a minimal **Git project** that uses Hugging Face `transformers` as a normal dependency (no debugging of the library), and shows how to load and run **dicta-il/dictabert-tiny-joint**.

## Quick start

```bash
# 1) Create & activate a virtual environment (recommended)
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

# 2) Install dependencies
pip install -r requirements.txt

# 3) Run the example
python main.py
```

If you prefer **Poetry**, see `pyproject.toml` below.

## What this includes

- `requirements.txt` → installs `transformers`, `torch`, `huggingface_hub`
- `main.py` → a simple script that loads the model and prints token‑level output
- `.vscode/settings.json` → VS Code picks `.venv` automatically
- `.gitignore` → ignores typical Python artifacts and your venv
- Optional `pyproject.toml` (Poetry users)

## Notes

- The model uses custom code from the Hub, so we pass `trust_remote_code=True`.
- You can select which heads to run (NER, morph, syntax, etc.).
- On first run it will download the model into your Hugging Face cache.

## Offline usage (optional)

After one successful online run, you can run offline by setting:

```bash
export TRANSFORMERS_OFFLINE=1           # macOS/Linux
set TRANSFORMERS_OFFLINE=1              # Windows PowerShell
```

Or download to a local folder with:

```bash
huggingface-cli download dicta-il/dictabert-tiny-joint --local-dir ./models/dictabert
```

Then in `main.py`, change the model path to `./models/dictabert`.

---
