#PDF text extraction via PyMuPDF
"""- Sentence-aware chunking
- TF-IDF based retrieval (no external API needed)
- Answer generation using a local sentence-transformers + extractive approach
  OR a local HuggingFace QA model (deepset/roberta-base-squad2)
"""

import re
import math
from collections import Counter
from typing import Any

def extract_text_from_pdf(path: str) -> tuple[str, int]:
    """Return (full_text, page_count) from a PDF file."""
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        pages = []
        for page in doc:
            pages.append(page.get_text())
        return "\n\n".join(pages), len(doc)
    except ImportError:
        raise RuntimeError("PyMuPDF (fitz) is required. Install with: pip install pymupdf")

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 80) -> list[str]:
    """Split text into overlapping word-level chunks."""
    # Normalise whitespace
    text = re.sub(r"\s+", " ", text).strip()
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk) > 50:          # skip tiny fragments
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


# TF-IDF retrieval

def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_idf(chunks: list[str]) -> dict[str, float]:
    N = len(chunks)
    df: Counter = Counter()
    for chunk in chunks:
        tokens = set(_tokenize(chunk))
        df.update(tokens)
    return {t: math.log((N + 1) / (freq + 1)) + 1 for t, freq in df.items()}


def _tfidf_score(query_tokens: list[str], chunk: str, idf: dict[str, float]) -> float:
    tf = Counter(_tokenize(chunk))
    total = max(sum(tf.values()), 1)
    score = 0.0
    for t in query_tokens:
        score += (tf.get(t, 0) / total) * idf.get(t, 0)
    return score


def retrieve_top_chunks(
    question: str,
    chunks: list[str],
    idf: dict[str, float],
    top_k: int = 4,
) -> list[tuple[str, float]]:
    q_tokens = _tokenize(question)
    scored = [(c, _tfidf_score(q_tokens, c, idf)) for c in chunks]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


# Extractive QA with a local HuggingFace model

_qa_pipeline = None

def _get_qa_pipeline():
    global _qa_pipeline
    if _qa_pipeline is None:
        try:
            from transformers import pipeline
            _qa_pipeline = pipeline(
                "question-answering",
                model="deepset/roberta-base-squad2",
                tokenizer="deepset/roberta-base-squad2",
            )
        except Exception as e:
            raise RuntimeError(f"Could not load QA model: {e}")
    return _qa_pipeline


def answer_with_model(question: str, context: str) -> dict[str, Any]:
    """Run extractive QA model over context."""
    pipe = _get_qa_pipeline()
    # HF pipeline handles long contexts by sliding window automatically
    result = pipe(question=question, context=context, max_answer_len=200)
    return {
        "answer": result["answer"],
        "score": round(float(result["score"]), 4),
    }


# Fallback: simple extractive heuristic (no model needed)

def extractive_fallback(question: str, context: str) -> dict[str, Any]:
    """Pick the sentence most similar to the question by token overlap."""
    sentences = re.split(r"(?<=[.!?])\s+", context)
    q_tokens = set(_tokenize(question))
    best_sent, best_score = "", 0.0
    for sent in sentences:
        s_tokens = set(_tokenize(sent))
        if not s_tokens:
            continue
        overlap = len(q_tokens & s_tokens) / math.sqrt(len(s_tokens))
        if overlap > best_score:
            best_score = overlap
            best_sent = sent
    return {"answer": best_sent or context[:300], "score": round(best_score, 4)}


# Public API

class DocumentQAEngine:
    def __init__(self, use_model: bool = True):
        self.use_model = use_model

    def process_document(self, pdf_path: str) -> dict[str, Any]:
        full_text, page_count = extract_text_from_pdf(pdf_path)
        chunks = chunk_text(full_text)
        idf = _build_idf(chunks)
        return {
            "full_text": full_text,
            "page_count": page_count,
            "chunks": chunks,
            "idf": idf,
        }

    def answer_question(self, question: str, doc_data: dict[str, Any]) -> dict[str, Any]:
        chunks = doc_data["chunks"]
        idf = doc_data["idf"]

        if not chunks:
            return {"answer": "The document appears to be empty.", "score": 0, "context": ""}

        top = retrieve_top_chunks(question, chunks, idf, top_k=4)
        context = " ".join(c for c, _ in top)

        if self.use_model:
            try:
                result = answer_with_model(question, context)
                method = "transformer"
            except Exception:
                result = extractive_fallback(question, context)
                method = "extractive_fallback"
        else:
            result = extractive_fallback(question, context)
            method = "extractive"

        return {
            "answer": result["answer"],
            "confidence": result["score"],
            "method": method,
            "context_used": context[:600] + ("…" if len(context) > 600 else ""),
            "relevant_chunks": [c[:300] for c, _ in top],
        }
