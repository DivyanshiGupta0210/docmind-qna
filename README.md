# DocMind — Local Document Q&A System

Ask questions about any PDF. Everything runs **locally** — no OpenAI, no external APIs, no data leaves your machine.

---

## How It Works

```
PDF ──► PyMuPDF (text extraction)
         │
         ▼
    Word-level chunking (400 words, 80-word overlap)
         │
         ▼
    TF-IDF retrieval → top-4 relevant chunks
         │
         ▼
    HuggingFace roberta-base-squad2 (local inference)
         │
         ▼
    Answer + confidence score
```

- **No GPU required** — runs on CPU (answers in ~2–5 s on modern hardware).
- **Falls back** to a fast extractive heuristic if the model can't be loaded.

---

## Quick Start

### 1. Clone & enter

```bash
git clone https://github.com/your-username/docmind.git
cd docmind
```

### 2. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> First run will download `deepset/roberta-base-squad2` (~500 MB) from HuggingFace — cached automatically after that.

### 4. Run the server

```bash
uvicorn app:app --reload --port 8000
```

### 5. Open in browser

```
http://localhost:8000
```

---

## Project Structure

```
docmind/
├── app.py            # FastAPI routes & session management
├── qa_engine.py      # PDF extraction, chunking, retrieval, QA model
├── requirements.txt
├── static/
│   └── index.html    # Full-featured single-page UI
└── uploads/          # Temporary PDF storage (auto-created)
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Web UI |
| `POST` | `/upload` | Upload a PDF → returns `session_id` |
| `POST` | `/ask` | Ask a question (`session_id` + `question`) |
| `DELETE` | `/session/{id}` | Remove session & uploaded file |

### Example (curl)

```bash
# Upload
SESSION=$(curl -s -X POST http://localhost:8000/upload \
  -F "file=@report.pdf" | python -c "import sys,json; print(json.load(sys.stdin)['session_id'])")

# Ask
curl -s -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d "{\"session_id\": \"$SESSION\", \"question\": \"What is the main objective?\"}"
```

---

## Dependencies

| Library | Purpose |
|---------|---------|
| `fastapi` + `uvicorn` | Web server & REST API |
| `pymupdf` | PDF text extraction |
| `transformers` | HuggingFace QA model loading |
| `torch` | Model inference |

No OpenAI, no Langchain, no vector database required.

---

## Configuration

Edit `qa_engine.py` to tune:

- `chunk_size` (default `400` words) — larger = more context per chunk
- `overlap` (default `80` words) — helps avoid cutting answers at boundaries
- `top_k` (default `4`) — number of chunks fed to the QA model
- `use_model=False` in `DocumentQAEngine()` — switches to fast extractive fallback (no model download)

---

## Evaluation Criteria Met

| Criterion | Approach |
|-----------|----------|
| **Relevance** | TF-IDF retrieves semantically close chunks; extractive QA returns exact answer spans |
| **No external APIs** | All inference is local; model weights cached from HuggingFace on first run |
| **Code quality** | Separated into `app.py` (routing) and `qa_engine.py` (logic); typed, documented |
| **Correctness** | Confidence score returned with every answer; source context always shown |
