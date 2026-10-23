"""Measure retrieval quality and (optionally) answer quality.

  python -m eval.evaluate                      # retrieval hit-rate and MRR, no API key needed
  python -m eval.evaluate --answers            # also grade LLM answers (needs ANTHROPIC_API_KEY)
"""
import argparse
import json
import os

from app.rag import RAGPipeline

p = argparse.ArgumentParser()
p.add_argument("--k", type=int, default=4)
p.add_argument("--answers", action="store_true", help="also grade generated answers")
p.add_argument("--backend", default=os.getenv("RETRIEVER", "tfidf"))
args = p.parse_args()

pipe = RAGPipeline("data/docs", args.backend)
with open("eval/questions.json") as f:
    questions = json.load(f)

hits_at_k, rr, kw_ok = 0, 0.0, 0
for q in questions:
    hits = pipe.retriever.search(q["question"], args.k)
    sources = [c.source for c, _ in hits]
    found = q["source"] in sources
    if found:
        hits_at_k += 1
        rr += 1 / (sources.index(q["source"]) + 1)
    if args.answers:
        ans = pipe.ask(q["question"], args.k)["answer"].lower()
        kw_ok += any(kw.lower() in ans for kw in q["keywords"])
    print(f"[{'OK ' if found else 'MISS'}] {q['question']}")

n = len(questions)
print(f"\nRetrieval hit@{args.k}: {hits_at_k}/{n} = {hits_at_k / n:.0%}")
print(f"Mean reciprocal rank: {rr / n:.2f}")
if args.answers:
    print(f"Answer keyword accuracy: {kw_ok}/{n} = {kw_ok / n:.0%}")
