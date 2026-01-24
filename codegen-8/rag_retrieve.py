#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retrieve RAG context from rag_index.sqlite.

Two modes:
- FTS-only (fast, no extra deps): bm25 ranking from SQLite FTS5.
- FTS + embeddings rerank: uses sentence-transformers to rerank top candidates.

The index schema must match rag_index_build.py.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from typing import Any, Dict, List, Tuple


def _connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA query_only=ON;")
    return conn


def _fts_escape(query: str) -> str:
    """
    FTS5 MATCH query is not SQL; it has its own syntax.
    We escape tokens and remove characters that break parsing (., :, etc.)
    """
    q = (query or "").strip()
    if not q:
        return ""
    # Replace punctuation that often breaks MATCH
    q = re.sub(r"[\.\:\;\(\)\[\]\{\}\=\+\-\*\/\\\|\"\'\`]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    # AND the tokens (safer than raw user text)
    toks = [t for t in q.split(" ") if t]
    return " AND ".join(toks[:30])


def fts_candidates(conn: sqlite3.Connection, db_id: str, query: str, candidates: int = 50) -> List[Dict[str, Any]]:
    q = _fts_escape(query)
    if not q:
        return []
    sql = """
    SELECT d.doc_id, d.db_id, d.doc_type, d.title, d.text, bm25(docs_fts) AS score
    FROM docs_fts
    JOIN docs d ON d.doc_id = docs_fts.doc_id
    WHERE docs_fts MATCH ? AND (d.db_id = ? OR d.db_id = '*')
    ORDER BY score
    LIMIT ?
    """
    rows = conn.execute(sql, (q, db_id, int(candidates))).fetchall()
    out = []
    for doc_id, dbid, dtype, title, text, score in rows:
        out.append({"doc_id": doc_id, "db_id": dbid, "doc_type": dtype, "title": title, "text": text, "score": float(-score)})
    return out


def _load_embeddings(conn: sqlite3.Connection, doc_ids: List[str]) -> Dict[str, Tuple[int, bytes]]:
    if not doc_ids:
        return {}
    qmarks = ",".join(["?"] * len(doc_ids))
    rows = conn.execute(f"SELECT doc_id, dim, vec FROM embeddings WHERE doc_id IN ({qmarks})", doc_ids).fetchall()
    return {r[0]: (int(r[1]), r[2]) for r in rows}


def retrieve(
    index_path: str,
    db_id: str,
    query: str,
    k: int = 5,
    candidates: int = 50,
    use_embeddings: bool = False,
    embed_model: str = "all-MiniLM-L6-v2",
) -> List[Dict[str, Any]]:
    conn = _connect(index_path)
    try:
        cands = fts_candidates(conn, db_id=db_id, query=query, candidates=candidates)
        if not cands:
            return []

        if not use_embeddings:
            return cands[:k]

        # rerank with embeddings
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except Exception:
            return cands[:k]

        doc_ids = [c["doc_id"] for c in cands]
        emb_map = _load_embeddings(conn, doc_ids)

        if not emb_map:
            return cands[:k]

        model = SentenceTransformer(embed_model)
        qv = model.encode([query], normalize_embeddings=True)
        qv = np.asarray(qv[0], dtype=np.float32)

        rescored = []
        for c in cands:
            doc_id = c["doc_id"]
            if doc_id not in emb_map:
                continue
            dim, vec = emb_map[doc_id]
            dv = np.frombuffer(vec, dtype=np.float32, count=dim)
            score = float(np.dot(qv, dv))
            cc = dict(c)
            cc["emb_score"] = score
            rescored.append(cc)

        rescored.sort(key=lambda x: x.get("emb_score", -1e9), reverse=True)
        return rescored[:k]
    finally:
        conn.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--db-id", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--candidates", type=int, default=50)
    ap.add_argument("--use-embeddings", action="store_true")
    ap.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    args = ap.parse_args()

    hits = retrieve(
        index_path=args.index,
        db_id=args.db_id,
        query=args.query,
        k=int(args.k),
        candidates=int(args.candidates),
        use_embeddings=bool(args.use_embeddings),
        embed_model=str(args.embed_model),
    )

    for i, h in enumerate(hits, 1):
        print(f"[{i}] score={h.get('emb_score', h.get('score')):.4f} doc_id={h['doc_id']} type={h['doc_type']} title={h['title']}")
        print(h["text"][:800].rstrip())
        print("-" * 80)


if __name__ == "__main__":
    main()
