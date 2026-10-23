"""Streamlit front end.  Run: streamlit run ui.py"""
import os
import streamlit as st
from app.rag import RAGPipeline

st.set_page_config(page_title="RAG Document Q&A", page_icon="📄")
st.title("📄 Document Q&A (RAG)")


@st.cache_resource
def load():
    return RAGPipeline(os.getenv("DOCS_DIR", "data/docs"), os.getenv("RETRIEVER", "tfidf"))


pipe = load()
st.caption(f"{len(pipe.chunks)} chunks indexed")
question = st.text_input("Ask a question about your documents")
k = st.slider("Passages to retrieve", 1, 8, 4)

if question:
    result = pipe.ask(question, k)
    st.subheader("Answer")
    st.write(result["answer"])
    st.subheader("Sources")
    for i, s in enumerate(result["sources"], 1):
        with st.expander(f"[{i}] {s['source']} (score {s['score']})"):
            st.write(s["text"])
