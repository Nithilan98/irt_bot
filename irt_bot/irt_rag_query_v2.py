"""
STEP 3 (v2): Query the IRT Knowledge Base (faster + clearer)

Improvements vs v1:
- Normalized embeddings for cosine search
- Adds simple score threshold option
- Uses OpenAI Responses API with configurable model

Run:
  python irt_rag_query_v2.py
  python irt_rag_query_v2.py --query "v2 dataset failed" --min-score 0.30
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

load_dotenv()

COLLECTION = "irt_knowledge_base"
EMBED_MODEL = "all-MiniLM-L6-v2"
STORAGE_DIR = "./qdrant_storage"
TOP_K = 5

ai = OpenAI()


def clean(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<@[A-Z0-9]+>", "", text)
    text = re.sub(r"<https?://[^>]+>", "[link]", text)
    return text.strip()


def search(query: str, embedder: SentenceTransformer, qclient: QdrantClient, top_k: int) -> list[dict[str, Any]]:
    vec = embedder.encode([query], normalize_embeddings=True)[0].tolist()
    # qdrant-client 1.x exposes search differently for HTTP vs local modes.
    # Local QdrantClient (used here) keeps the low-level client on `. _client`.
    if hasattr(qclient, "search"):
        results = qclient.search(
            collection_name=COLLECTION,
            query_vector=vec,
            limit=top_k,
            with_payload=True,
        )
    else:
        # Fallback for local client where search lives on the inner client.
        inner = getattr(qclient, "_client", qclient)
        results = inner.search(
            collection_name=COLLECTION,
            query_vector=vec,
            limit=top_k,
            with_payload=True,
        )
    out: list[dict[str, Any]] = []
    for r in results:
        out.append(
            {
                "score": float(r.score),
                "summary": r.payload.get("summary", ""),
                "solution": clean(r.payload.get("solution", "")),
                "resolution_status": r.payload.get("resolution_status", r.payload.get("status", "")),
                "severity": r.payload.get("severity", ""),
                "bug_category": r.payload.get("bug_category", ""),
                "environment": r.payload.get("environment", ""),
                "references": r.payload.get("references", "None"),
                "team": r.payload.get("team", ""),
            }
        )
    return out


def generate_answer(query: str, hits: list[dict[str, Any]]) -> str:
    context = ""
    for i, h in enumerate(hits, 1):
        context += f"""
Previous ticket #{i} (relevance: {h['score']:.3f})
  Summary        : {h['summary']}
  Category       : {h['bug_category']}
  Severity       : {h['severity']}
  Environment    : {h['environment']}
  Final Status   : {h['resolution_status']}
  Solution notes : {h['solution']}
  References     : {h['references']}
  Team           : {h['team']}
"""

    resp = ai.responses.create(
        model=os.environ.get("OPENAI_MODEL_ANSWER", "gpt-4.1"),
        max_output_tokens=900,
        input=f"""
You are an IRT (Incident Response Team) support assistant for ConverSight.
You are answering a question from an IRT team member or a client who has raised a support issue.

The person asked:
"{query}"

Here are the most relevant cases the IRT team has handled before:
{context}

Rules for your answer:
1. ALWAYS base your answer on the actual previous cases above — do NOT give generic advice like "refresh the page" or "log out and log back in" unless that was the actual fix.
2. If the previous cases show a real fix or workaround, explain it clearly and specifically (e.g. "In a previous similar case, the fix was to reduce the number of entities by changing non-ID columns to ID columns").
3. Give 2–4 specific things to check or try, based on what actually worked before.
4. Use simple language — avoid deep programming words, but it is okay to say things like "dataset", "entity count", "SME publish", "vacuum", "republish" — these are normal IRT terms.
5. If multiple root causes have been seen before, mention each one briefly with what was done to fix it.
6. End with: "If the above steps do not help, please contact the IRT team with your dataset name, org ID, and environment."
7. Do NOT say "knowledge base", "previous tickets", or "Issue #1". Just say "In similar cases..." or "The IRT team has seen this before...".
8. Keep the answer under 350 words. Use bullet points for steps.
""",
    )
    return (resp.output_text or "").strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", "-q", type=str, help="Single query to run")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--min-score", type=float, default=0.0, help="If top hit is below this, treat as not found")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("❌ OPENAI_API_KEY missing (export it or set it in .env)")

    print("  Loading embedding model …")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("  Connecting to Qdrant …")
    qclient = QdrantClient(path=STORAGE_DIR)

    existing = [c.name for c in qclient.get_collections().collections]
    if COLLECTION not in existing:
        print(f"❌ Collection '{COLLECTION}' not found!")
        print("   Run irt_rag_build_knowledge_base.py (or v2) first.")
        return

    count = qclient.count(collection_name=COLLECTION).count
    print(f"  ✅ Knowledge base ready: {count} documents\n")

    def run_one(q: str) -> None:
        print(f"\n{'='*60}")
        print(f"  Query: {q}")
        print("=" * 60)

        hits = search(q, embedder, qclient, args.top_k)
        if not hits or (args.min_score and hits[0]["score"] < args.min_score):
            print("  ❌ No similar issues found in knowledge base.")
            return

        print("  🤖 AI Answer:\n")
        print(generate_answer(q, hits))
        print()

    if args.query:
        run_one(args.query)
        return

    print("=" * 60)
    print("  🐛 IRT Knowledge Base — Interactive Query (v2)")
    print("=" * 60)
    print("  Type your question and press Enter.")
    print("  Type 'quit' to exit.\n")

    while True:
        try:
            q = input("❓ Question: ").strip()
            if not q:
                continue
            if q.lower() in ("quit", "exit", "q"):
                print("Bye!")
                break
            run_one(q)
        except KeyboardInterrupt:
            print("\nBye!")
            break


if __name__ == "__main__":
    main()