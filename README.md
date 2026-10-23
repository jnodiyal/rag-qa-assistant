# RAG Document Q&A Assistant

A retrieval-augmented generation (RAG) system that answers questions over your own documents and cites its sources.
Built with **Python, scikit-learn (TF-IDF) or sentence-transformers + FAISS, FastAPI, Streamlit, and an LLM API (Anthropic Claude)**.

## How it works
1. **Load** `.txt`, `.md`, and `.pdf` files from `data/docs/`
2. **Chunk** each document into overlapping 120-word windows (30-word overlap)
3. **Index** chunks as vectors (TF-IDF by default; `RETRIEVER=embed` switches to sentence-transformer embeddings + FAISS)
4. **Retrieve** the top-k most similar chunks for a question
5. **Generate** an answer from only those chunks with source citations (falls back to showing the best passage if no API key is set)
6. **Serve** through a FastAPI REST API and a Streamlit UI
7. **Evaluate** retrieval hit-rate and MRR on a test question set

## Quick start
```bash
pip install -r requirements.txt
python -m eval.evaluate                  # retrieval evaluation (no API key needed)
uvicorn app.api:app --reload             # API at http://127.0.0.1:8000/docs
streamlit run ui.py                      # web UI
```
For LLM-written answers: `export ANTHROPIC_API_KEY=your_key` (optional `ANTHROPIC_MODEL=...`).
To use your own documents, drop files into `data/docs/`.

## API
`POST /ask` with `{"question": "What does a data steward do?", "k": 4}` returns the answer plus the source chunks and scores. `GET /health` reports how many chunks are indexed.

## Evaluation
`python -m eval.evaluate` runs 12 test questions (`eval/questions.json`) and reports hit@k and mean reciprocal rank.
`python -m eval.evaluate --answers` (needs an API key) also checks generated answers for expected keywords.

| Retriever | hit@4 | MRR |
|---|---|---|
| TF-IDF, no stemming | 92% (11/12) | 0.92 |
| TF-IDF + light stemmer | 100% (12/12) | 1.00 |

Note: the sample corpus is small (4 documents), so scores are high. Add larger documents and harder questions to stress-test it.
The first run missed "evaluate forecasts" because the documents say "forecasting" and "error metrics"; adding a stemmer fixed it, which shows why embeddings (`RETRIEVER=embed`) matter on larger corpora.

## Project structure
```
app/rag.py       loading, chunking, retrieval, generation
app/api.py       FastAPI service
ui.py            Streamlit front end
eval/            test questions and evaluation script
data/docs/       sample documents
```

## Demo

![RAG QA Assistant](assets/rag-qa-demo.png)
![FastAPI API Documentation](api-docs.png)
## Ideas for next steps
Re-ranking, hybrid (keyword + vector) search, a React front end, conversation memory, Docker deployment on AWS.
