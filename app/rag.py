"""Core RAG pipeline: load -> chunk -> index -> retrieve -> generate."""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------- loading
def load_documents(folder: str | Path) -> list[dict]:
    """Read .txt / .md / .pdf files into [{'source', 'text'}]."""
    docs = []
    for path in sorted(Path(folder).glob("**/*")):
        if path.suffix.lower() in {".txt", ".md"}:
            text = path.read_text(encoding="utf-8", errors="ignore")
        elif path.suffix.lower() == ".pdf":
            from pypdf import PdfReader
            text = "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
        else:
            continue
        if text.strip():
            docs.append({"source": path.name, "text": text})
    return docs


# ---------------------------------------------------------------- chunking
def chunk_text(text: str, chunk_words: int = 120, overlap: int = 30) -> list[str]:
    """Split text into overlapping word windows."""
    words = text.split()
    step = max(1, chunk_words - overlap)
    chunks = []
    for start in range(0, len(words), step):
        piece = words[start:start + chunk_words]
        if piece:
            chunks.append(" ".join(piece))
        if start + chunk_words >= len(words):
            break
    return chunks


@dataclass
class Chunk:
    source: str
    text: str


def build_chunks(docs: list[dict], **kw) -> list[Chunk]:
    return [Chunk(d["source"], c) for d in docs for c in chunk_text(d["text"], **kw)]


# --------------------------------------------------------------- retrieval
_SUFFIXES = ("ations", "ation", "ings", "ing", "ies", "ed", "es", "s")


def tokenize(text: str) -> list[str]:
    """Lowercase word tokens with a light suffix-stripping stemmer
    (so 'forecasts' and 'forecasting' both match 'forecast')."""
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        for suf in _SUFFIXES:
            if w.endswith(suf) and len(w) - len(suf) >= 4:
                w = w[: -len(suf)]
                break
        out.append(w)
    return out


class Retriever:
    """Vector search over chunks.

    backend="tfidf": no downloads, works offline (default).
    backend="embed": sentence-transformers embeddings (+ FAISS if installed).
    """

    def __init__(self, chunks: list[Chunk], backend: str = "tfidf"):
        self.chunks = chunks
        self.backend = backend
        texts = [c.text for c in chunks]
        if backend == "embed":
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.matrix = self.model.encode(texts, normalize_embeddings=True).astype("float32")
            try:
                import faiss
                self.index = faiss.IndexFlatIP(self.matrix.shape[1])
                self.index.add(self.matrix)
            except ImportError:
                self.index = None
        else:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self.vec = TfidfVectorizer(
                tokenizer=tokenize, token_pattern=None, ngram_range=(1, 2), sublinear_tf=True
            )
            self.matrix = self.vec.fit_transform(texts)

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        if self.backend == "embed":
            q = self.model.encode([query], normalize_embeddings=True).astype("float32")
            if self.index is not None:
                scores, idx = self.index.search(q, k)
                pairs = list(zip(idx[0], scores[0]))
            else:
                sims = (self.matrix @ q.T).ravel()
                top = np.argsort(-sims)[:k]
                pairs = [(i, sims[i]) for i in top]
        else:
            q = self.vec.transform([query])
            sims = (self.matrix @ q.T).toarray().ravel()
            top = np.argsort(-sims)[:k]
            pairs = [(i, sims[i]) for i in top]
        return [(self.chunks[int(i)], float(s)) for i, s in pairs if s > 0]


# -------------------------------------------------------------- generation
PROMPT = """You are a careful assistant. Answer the question using ONLY the context below.
Cite the sources you used like [1], [2]. If the context does not contain the answer,
say "I don't know based on the provided documents."

Context:
{context}

Question: {question}
Answer:"""


def generate_answer(question: str, hits: list[tuple[Chunk, float]]) -> str:
    """Call an LLM if ANTHROPIC_API_KEY is set, else return the top passage."""
    if not hits:
        return "I don't know based on the provided documents."
    context = "\n\n".join(f"[{i}] ({c.source}) {c.text}" for i, (c, _) in enumerate(hits, 1))
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        best = hits[0][0]
        return f"(No API key set - showing best matching passage from {best.source})\n{best.text}"
    import anthropic
    client = anthropic.Anthropic(api_key=key)
    msg = client.messages.create(
        model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5"),
        max_tokens=500,
        messages=[{"role": "user", "content": PROMPT.format(context=context, question=question)}],
    )
    return "".join(b.text for b in msg.content if b.type == "text").strip()


class RAGPipeline:
    def __init__(self, docs_dir: str = "data/docs", backend: str = "tfidf"):
        self.chunks = build_chunks(load_documents(docs_dir))
        if not self.chunks:
            raise ValueError(f"No documents found in {docs_dir}")
        self.retriever = Retriever(self.chunks, backend)

    def ask(self, question: str, k: int = 4) -> dict:
        hits = self.retriever.search(question, k)
        return {
            "question": question,
            "answer": generate_answer(question, hits),
            "sources": [
                {"source": c.source, "score": round(s, 3), "text": c.text} for c, s in hits
            ],
        }
