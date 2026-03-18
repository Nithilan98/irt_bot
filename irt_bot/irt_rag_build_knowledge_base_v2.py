"""
STEP 2 (v2): Build or update Qdrant Knowledge Base (faster + incremental)

Improvements vs v1:
- Uses embedder's native dimension (no hardcoded VECTOR_DIM)
- Normalizes embeddings for cosine similarity
- Supports incremental updates (default) or full recreate (--recreate)
- Stores a stable point ID so reruns don't duplicate points

Input : bugs_enriched.xlsx
Output: local qdrant_storage/ (Qdrant local mode)

Run:
  python irt_rag_build_knowledge_base_v2.py
  python irt_rag_build_knowledge_base_v2.py --recreate
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import uuid
from typing import Any

import pandas as pd
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer

load_dotenv()

INPUT_FILE = "bugs_enriched.xlsx"
COLLECTION = "irt_knowledge_base"
EMBED_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 96
STORAGE_DIR = "./qdrant_storage"


def clean(text: Any) -> str:
    if not isinstance(text, str) or text in ("nan", "None", ""):
        return ""
    text = re.sub(r"<@[A-Z0-9]+>", "", text)
    text = re.sub(r"<https?://[^>]+>", "[link]", text)
    return text.strip()


def build_document(row: pd.Series) -> str:
    parts: list[str] = []
    summary = str(row.get("Summary", "")).strip()
    comments = clean(str(row.get("Comments", "")))
    solution = clean(str(row.get("Solution", "")))
    category = str(row.get("Bug Category", "")).strip()
    env = str(row.get("Environment", "")).strip()
    severity = str(row.get("Severity", "")).strip()

    if summary:
        parts.append(f"Issue: {summary}")
    if category and category != "nan":
        parts.append(f"Category: {category}")
    if env and env != "nan":
        parts.append(f"Environment: {env}")
    if severity and severity != "nan":
        parts.append(f"Severity: {severity}")
    if comments:
        parts.append(f"Discussion: {comments[:900]}")
    if solution:
        parts.append(f"Solution: {solution[:900]}")

    return "\n".join(parts).strip()


def stable_point_id(row: pd.Series) -> str:
    # Deterministic, but must be a valid UUID string for local Qdrant.
    key = (
        str(row.get("Summary", "")).strip()
        + "|"
        + str(row.get("Comments", "")).strip()
        + "|"
        + str(row.get("Date submitted", "")).strip()
    )
    digest = hashlib.sha1(key.encode("utf-8", errors="ignore")).hexdigest()
    # Use first 32 hex chars as UUID5-style namespace; this is stable per key.
    return str(uuid.UUID(hex=digest[:32]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_FILE)
    parser.add_argument("--storage", default=STORAGE_DIR)
    parser.add_argument("--collection", default=COLLECTION)
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("KB_BATCH_SIZE", str(BATCH_SIZE))))
    parser.add_argument("--recreate", action="store_true", help="Delete and rebuild collection from scratch")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  STEP 2 (v2) — Build/Update Qdrant Knowledge Base")
    print("=" * 60)

    if not os.path.exists(args.input):
        raise SystemExit(f"❌ {args.input} not found. Run enrichment first.")

    df = pd.read_excel(args.input)
    print(f"  Loaded: {len(df)} rows from {args.input}")

    resolution_col = "Resolution Status" if "Resolution Status" in df.columns else ("status" if "status" in df.columns else None)
    if not resolution_col:
        raise SystemExit("❌ Missing column: 'Resolution Status' (or 'status'). Run enrichment first.")

    useful = df[df[resolution_col].isin(["Fixed", "Partial"])].copy()
    print(f"  Rows to index (Fixed/Partial): {len(useful)}")

    print(f"\n  Loading embedding model: {EMBED_MODEL} …")
    embedder = SentenceTransformer(EMBED_MODEL)
    dim = int(getattr(embedder, "get_sentence_embedding_dimension")())
    print(f"  ✅ Embedding model ready (dim={dim})")

    print(f"\n  Connecting to local Qdrant: {args.storage}")
    qclient = QdrantClient(path=args.storage)

    existing = {c.name for c in qclient.get_collections().collections}
    if args.recreate and args.collection in existing:
        qclient.delete_collection(args.collection)
        existing.remove(args.collection)
        print(f"  Deleted old collection '{args.collection}'")

    if args.collection not in existing:
        qclient.create_collection(
            collection_name=args.collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        print(f"  ✅ Collection '{args.collection}' created (cosine)")

    batch_docs: list[str] = []
    batch_payloads: list[dict[str, Any]] = []
    batch_ids: list[str] = []

    total = 0
    target = len(useful)
    bs = max(1, args.batch_size)

    print(f"\n  Upserting {target} documents (batch size={bs}) …\n")
    for _, row in useful.iterrows():
        doc = build_document(row)
        if not doc:
            continue

        pid = stable_point_id(row)
        payload = {
            "summary": str(row.get("Summary", ""))[:250],
            "solution": str(row.get("Solution", ""))[:1200],
            "resolution_status": str(row.get(resolution_col, "")),
            "status": str(row.get(resolution_col, "")),
            "severity": str(row.get("Severity", "")),
            "bug_category": str(row.get("Bug Category", "")),
            "environment": str(row.get("Environment", "")),
            "references": str(row.get("References", "")),
            "team": str(row.get("Team/Department", "")),
            "assignee": str(row.get("Assignee", "")),
            "date_submitted": str(row.get("Date submitted", "")),
            "doc_text": doc[:1200],
        }

        batch_docs.append(doc)
        batch_payloads.append(payload)
        batch_ids.append(pid)

        if len(batch_docs) >= bs:
            vectors = embedder.encode(batch_docs, normalize_embeddings=True, show_progress_bar=False).tolist()
            points = [PointStruct(id=i, vector=v, payload=p) for i, v, p in zip(batch_ids, vectors, batch_payloads)]
            qclient.upsert(collection_name=args.collection, points=points)
            total += len(points)
            print(f"  Upserted: {total}/{target}")
            batch_docs, batch_payloads, batch_ids = [], [], []

    if batch_docs:
        vectors = embedder.encode(batch_docs, normalize_embeddings=True, show_progress_bar=False).tolist()
        points = [PointStruct(id=i, vector=v, payload=p) for i, v, p in zip(batch_ids, vectors, batch_payloads)]
        qclient.upsert(collection_name=args.collection, points=points)
        total += len(points)

    final_count = qclient.count(collection_name=args.collection).count
    print()
    print("=" * 60)
    print("  ✅ Knowledge base ready")
    print("=" * 60)
    print(f"  Collection: {args.collection}")
    print(f"  Stored    : {final_count}")
    print(f"  Storage   : {args.storage}/")
    print()


if __name__ == "__main__":
    main()

