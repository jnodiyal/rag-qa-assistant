"""FastAPI service.  Run: uvicorn app.api:app --reload"""
import os

from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import RAGPipeline

app = FastAPI(title="RAG Document Q&A API")
pipeline = RAGPipeline(
    docs_dir=os.getenv("DOCS_DIR", "data/docs"),
    backend=os.getenv("RETRIEVER", "tfidf"),
)


class Query(BaseModel):
    question: str
    k: int = 4


@app.get("/health")
def health():
    return {"status": "ok", "chunks_indexed": len(pipeline.chunks)}


@app.post("/ask")
def ask(q: Query):
    return pipeline.ask(q.question, q.k)
