#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rag_retrieve_fast_v2.py

FAST retrieval for rag_index.sqlite with:
- Thread-local sqlite connection (no reconnect)
- Read-optimized PRAGMAs
- Safe FTS query (drops numeric-only tokens like "5")
- Optional embedding rerank that uses SHORT SNIPPETS (not full text) to avoid 80s slowdowns
- Cached SentenceTransformer model (loaded once per process)

API:
    hits = retrieve(index_path, db_id, query, k=6, candidates=40, use_embeddings=False, embed_model="all-MiniLM-L6-v2")
Each hit: {source, doc_type, title, text, score, fts_query, (emb_score)}
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import threading
from functools import lru_cache
from typing import Any, Dict, List, Optional

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except Exception:
    SentenceTransformer = None
    np = None

# Keep small; this is schema retrieval not long-doc semantic search
_EMBED_SNIPPET_CHARS = 700
_MAX_RERANK = 24  # rerank only first N candidates
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")

STOPWORDS = {
    "a","an","and","are","as","at","be","by","for","from","has","have","how","i","in","is","it","its","me","my",
    "of","on","or","that","the","their","then","there","these","this","those","to","was","were","what","when",
    "where","which","who","why","with","will","would","show","list","give","find","top","highest","lowest","most",
    "least","get","please","all",
}

_TLS = threading.local()

def _get_conn(index_path: str) -> sqlite3.Connection:
    conn = getattr(_TLS, "conn", None)
    path = getattr(_TLS, "path", None)
    if conn is not None and path == index_path:
        return conn

    conn = sqlite3.connect(index_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        conn.execute("PRAGMA temp_store=MEMORY;")
        conn.execute("PRAGMA cache_size=-200000;")  # ~200MB
    except Exception:
        pass
    _TLS.conn = conn
    _TLS.path = index_path
    return conn

def _tokenize(query: str) -> List[str]:
    q = (query or "").lower()
    toks = _WORD_RE.findall(q)
    out: List[str] = []
    for t in toks:
        if not t:
            continue
        if t in STOPWORDS:
            continue
        # drop numeric-only tokens (this kills "5*" in fts_query)
        if t.isdigit():
            continue
        out.append(t)
    return out[:24]

def build_fts_query(query: str) -> str:
    toks = _tokenize(query)
    if not toks:
        return ""
    # prefix OR query: token* OR token*
    return " OR ".join(f"{t}*" for t in toks)

def fts_candidates(conn: sqlite3.Connection, db_id: str, query: str, candidates: int = 40) -> List[Dict[str, Any]]:
    fts_query = build_fts_query(query)
    if not fts_query:
        return []

    rows = conn.execute(
        """
        SELECT d.doc_id, d.db_id, d.source, d.doc_type, d.title, d.text, bm25(docs_fts) AS bm25
        FROM docs_fts
        JOIN docs d ON d.doc_id = docs_fts.doc_id
        WHERE docs_fts MATCH ? AND (d.db_id = ? OR d.db_id='*')
        ORDER BY bm25
        LIMIT ?
        """,
        (fts_query, db_id, int(candidates)),
    ).fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "doc_id": r["doc_id"],
                "db_id": r["db_id"],
                "source": r["source"],
                "doc_type": r["doc_type"],
                "title": r["title"],
                "text": r["text"],
                "score": float(-r["bm25"]),  # lower bm25 is better -> invert
                "fts_query": fts_query,
            }
        )
    return out

def like_fallback(conn: sqlite3.Connection, db_id: str, query: str, limit: int = 30) -> List[Dict[str, Any]]:
    toks = _tokenize(query)
    if not toks:
        return []
    pat = "%" + "%".join(toks[:6]) + "%"
    rows = conn.execute(
        """
        SELECT doc_id, db_id, source, doc_type, title, text
        FROM docs
        WHERE (db_id = ? OR db_id='*')
          AND (title LIKE ? OR text LIKE ?)
        LIMIT ?
        """,
        (db_id, pat, pat, int(limit)),
    ).fetchall()

    out: List[Dict[str, Any]] = []
    for r in rows:
        out.append(
            {
                "doc_id": r["doc_id"],
                "db_id": r["db_id"],
                "source": r["source"],
                "doc_type": r["doc_type"],
                "title": r["title"],
                "text": r["text"],
                "score": 0.0,
                "fts_query": "",
            }
        )
    return out

@lru_cache(maxsize=2)
def _get_embedder(model_name: str):
    if SentenceTransformer is None:
        raise RuntimeError("sentence-transformers not installed")
    return SentenceTransformer(model_name)

def _snippet_for_embed(hit: Dict[str, Any]) -> str:
    # Use title + first N chars of text; embedding full text is slow.
    title = (hit.get("title") or "").strip()
    text = (hit.get("text") or "").strip()
    if len(text) > _EMBED_SNIPPET_CHARS:
        text = text[:_EMBED_SNIPPET_CHARS]
    if title:
        return f"{title}\n{text}"
    return text

def embed_rerank(embed_model: str, query: str, cands: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    if not cands:
        return []
    if np is None:
        return cands[:top_k]

    embedder = _get_embedder(embed_model)
    subset = cands[: min(len(cands), max(_MAX_RERANK, top_k * 4))]

    qv = embedder.encode([query], normalize_embeddings=True)[0].astype("float32")
    dv = embedder.encode([_snippet_for_embed(c) for c in subset], normalize_embeddings=True).astype("float32")

    scores = (dv @ qv).tolist()
    rescored = []
    for c, s in zip(subset, scores):
        cc = dict(c)
        cc["emb_score"] = float(s)
        rescored.append(cc)
    rescored.sort(key=lambda x: x.get("emb_score", -1e9), reverse=True)

    # Fill with remaining (FTS order) if needed
    tail = cands[len(subset):]
    return rescored[:top_k] + tail[: max(0, top_k - len(rescored))]

def retrieve(
    index_path: str,
    db_id: str,
    query: str,
    k: int = 6,
    candidates: int = 40,
    use_embeddings: bool = False,
    embed_model: str = "all-MiniLM-L6-v2",
) -> List[Dict[str, Any]]:
    conn = _get_conn(index_path)
    cands = fts_candidates(conn, db_id=db_id, query=query, candidates=candidates)
    if not cands:
        cands = like_fallback(conn, db_id=db_id, query=query, limit=candidates)
    if not cands:
        return []
    if use_embeddings:
        try:
            return embed_rerank(embed_model, query, cands, top_k=int(k))
        except Exception:
            return cands[:k]
    return cands[:k]

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--db-id", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--candidates", type=int, default=40)
    ap.add_argument("--use-embeddings", action="store_true")
    ap.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    args = ap.parse_args()

    import time
    t0 = time.time()
    hits = retrieve(
        index_path=args.index,
        db_id=args.db_id,
        query=args.query,
        k=args.k,
        candidates=args.candidates,
        use_embeddings=args.use_embeddings,
        embed_model=args.embed_model,
    )
    dt = time.time() - t0
    print(f"hits={len(hits)} in {dt:.3f}s")
    if hits and hits[0].get("fts_query"):
        print("fts_query:", hits[0]["fts_query"])
    for h in hits:
        print(f"- {h.get('source')} | {h.get('doc_type')} | {h.get('title')} | score={h.get('emb_score', h.get('score'))}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
