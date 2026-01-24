#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a tiny RAG index for BIRD schemas + rules into a single SQLite DB.

Schema (consistent across tools):
- docs(doc_id TEXT PRIMARY KEY, db_id TEXT, doc_type TEXT, title TEXT, text TEXT, source TEXT, meta_json TEXT)
- docs_fts: FTS5 over (title, text) with doc_id + db_id as UNINDEXED columns
- embeddings(doc_id TEXT PRIMARY KEY, dim INTEGER, vec BLOB)

Notes:
- NEVER CREATE a normal index on an FTS virtual table (SQLite forbids it).
- --reset deletes the output sqlite file (prevents "database malformed" + stale schema issues).
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from tqdm import tqdm


RULE_DOCS: List[Dict[str, Any]] = [
    {
        "doc_id": "rule:basic-operators",
        "db_id": "*",
        "doc_type": "rules",
        "title": "Mongo aggregation guardrails",
        "text": (
            "Always produce MongoDB aggregation pipelines.\n"
            "Every pipeline stage must be an object with exactly one key, like {$match:{...}}.\n"
            "Avoid unsupported operators: $date, $currentDate, $dateConstant.\n"
            "Prefer $dateFromString for parsing strings; use $toDate only if the field is ISO string.\n"
            "For joins: use $lookup then $unwind (preserveNullAndEmptyArrays:false) when filtering on joined fields.\n"
            "Always end with $project selecting ONLY the SQL SELECT columns (and exclude _id unless needed).\n"
        ),
        "source": "built-in",
        "meta_json": "{}",
    },
    {
        "doc_id": "rule:join-keys",
        "db_id": "*",
        "doc_type": "rules",
        "title": "Join key reminders",
        "text": (
            "Do NOT assume _id is the primary key. Use the actual schema primary keys like movie_id, user_id, list_id.\n"
            "When joining tables, match on the columns used in the SQL ON clause.\n"
        ),
        "source": "built-in",
        "meta_json": "{}",
    },
    {
        "doc_id": "rule:count-avg",
        "db_id": "*",
        "doc_type": "rules",
        "title": "Counts and averages",
        "text": (
            "COUNT(*) -> {$count:'count'} or group with {$sum:1}.\n"
            "AVG(x) -> {$group:{_id:null, avg:{$avg:'$x'}}}.\n"
            "If SQL returns a single scalar, output a single document with one field (and optional _id:null).\n"
        ),
        "source": "built-in",
        "meta_json": "{}",
    },
    {
        "doc_id": "rule:dates",
        "db_id": "*",
        "doc_type": "rules",
        "title": "Date filters",
        "text": (
            "If SQL uses LIKE '2020%' or strftime('%Y', col)='2020', prefer regex ^2020 on the string column.\n"
            "Do not invent date operators.\n"
        ),
        "source": "built-in",
        "meta_json": "{}",
    },
    {
        "doc_id": "rule:strict-project",
        "db_id": "*",
        "doc_type": "rules",
        "title": "Strict projection",
        "text": (
            "SQL SELECT columns must map to $project output fields.\n"
            "Example: SELECT movies.movie_title -> {$project:{_id:0, movie_title:1}}.\n"
        ),
        "source": "built-in",
        "meta_json": "{}",
    },
]


def connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS docs(
            doc_id TEXT PRIMARY KEY,
            db_id TEXT NOT NULL,
            doc_type TEXT NOT NULL,
            title TEXT NOT NULL,
            text TEXT NOT NULL,
            source TEXT NOT NULL,
            meta_json TEXT NOT NULL DEFAULT '{}'
        );

        -- FTS virtual table (cannot be indexed with CREATE INDEX)
        CREATE VIRTUAL TABLE IF NOT EXISTS docs_fts USING fts5(
            doc_id UNINDEXED,
            db_id UNINDEXED,
            title,
            text,
            tokenize='unicode61'
        );

        CREATE TABLE IF NOT EXISTS embeddings(
            doc_id TEXT PRIMARY KEY,
            dim INTEGER NOT NULL,
            vec BLOB NOT NULL,
            FOREIGN KEY(doc_id) REFERENCES docs(doc_id) ON DELETE CASCADE
        );
        """
    )
    conn.commit()


def reset_index(path: str) -> None:
    if os.path.exists(path):
        os.remove(path)


def load_train_tables(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def schema_docs(train_tables: Dict[str, Any]) -> List[Dict[str, Any]]:
    docs = []
    # train_tables is typically a list of db schema dicts
    for db in train_tables:
        db_id = db.get("db_id") or db.get("db") or db.get("database") or ""
        if not db_id:
            continue

        # tables + columns
        table_names = db.get("table_names_original") or db.get("table_names") or []
        col_names = db.get("column_names_original") or db.get("column_names") or []
        col_types = db.get("column_types") or []
        pk = db.get("primary_keys") or []
        fk = db.get("foreign_keys") or []

        # build per-table text
        tables: Dict[int, Dict[str, Any]] = {i: {"name": table_names[i], "cols": []} for i in range(len(table_names))}
        for i, col in enumerate(col_names):
            # col is usually [table_idx, col_name]
            if not isinstance(col, (list, tuple)) or len(col) != 2:
                continue
            t_idx, c_name = col
            if t_idx == -1:
                continue
            c_type = col_types[i] if i < len(col_types) else ""
            tables[t_idx]["cols"].append((c_name, c_type, i in pk))

        fk_pairs = []
        for pair in fk:
            if isinstance(pair, (list, tuple)) and len(pair) == 2:
                fk_pairs.append(pair)

        # per-db overview doc
        overview_lines = [f"DB: {db_id}", ""]
        for t_idx, info in tables.items():
            overview_lines.append(f"Table {info['name']}:")
            for (c_name, c_type, is_pk) in info["cols"]:
                overview_lines.append(f"  - {c_name} ({c_type}){' [PK]' if is_pk else ''}")
            overview_lines.append("")
        if fk_pairs:
            overview_lines.append("Foreign keys (by column index):")
            for a, b in fk_pairs:
                overview_lines.append(f"  - {a} -> {b}")

        docs.append(
            {
                "doc_id": f"schema:{db_id}:overview",
                "db_id": db_id,
                "doc_type": "schema",
                "title": f"{db_id} schema overview",
                "text": "\n".join(overview_lines).strip(),
                "source": "train_tables.json",
                "meta_json": json.dumps({"db_id": db_id}),
            }
        )

        # per-table docs (better retrieval)
        for t_idx, info in tables.items():
            lines = [f"DB: {db_id}", f"Table: {info['name']}", ""]
            for (c_name, c_type, is_pk) in info["cols"]:
                lines.append(f"- {c_name} ({c_type}){' [PK]' if is_pk else ''}")
            docs.append(
                {
                    "doc_id": f"schema:{db_id}:table:{info['name']}",
                    "db_id": db_id,
                    "doc_type": "schema_table",
                    "title": f"{db_id}.{info['name']} columns",
                    "text": "\n".join(lines),
                    "source": "train_tables.json",
                    "meta_json": json.dumps({"db_id": db_id, "table": info["name"]}),
                }
            )

    return docs


def upsert_docs(conn: sqlite3.Connection, docs: List[Dict[str, Any]]) -> None:
    conn.executemany(
        """
        INSERT OR REPLACE INTO docs(doc_id, db_id, doc_type, title, text, source, meta_json)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                d["doc_id"],
                d["db_id"],
                d["doc_type"],
                d["title"],
                d["text"],
                d.get("source", ""),
                d.get("meta_json", "{}"),
            )
            for d in docs
        ],
    )

    # keep fts in sync (simple rebuild of fts rows for these doc_ids)
    conn.executemany("DELETE FROM docs_fts WHERE doc_id = ?", [(d["doc_id"],) for d in docs])
    conn.executemany(
        """
        INSERT INTO docs_fts(doc_id, db_id, title, text)
        VALUES (?, ?, ?, ?)
        """,
        [(d["doc_id"], d["db_id"], d["title"], d["text"]) for d in docs],
    )
    conn.commit()


def build_embeddings(conn: sqlite3.Connection, docs: List[Dict[str, Any]], model_name: str, batch_size: int = 32) -> None:
    from sentence_transformers import SentenceTransformer
    import numpy as np

    model = SentenceTransformer(model_name)

    # encode title+text for better semantics
    texts = [f"{d['title']}\n{d['text']}" for d in docs]
    ids = [d["doc_id"] for d in docs]

    # clear existing embeddings for these docs
    conn.executemany("DELETE FROM embeddings WHERE doc_id = ?", [(i,) for i in ids])
    conn.commit()

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch_texts = texts[i : i + batch_size]
        batch_ids = ids[i : i + batch_size]
        vecs = model.encode(batch_texts, normalize_embeddings=True)
        vecs = np.asarray(vecs, dtype=np.float32)
        dim = int(vecs.shape[1])
        rows = [(doc_id, dim, vecs[j].tobytes()) for j, doc_id in enumerate(batch_ids)]
        conn.executemany("INSERT OR REPLACE INTO embeddings(doc_id, dim, vec) VALUES (?, ?, ?)", rows)
        conn.commit()


def stats(conn: sqlite3.Connection) -> Tuple[int, int, int]:
    d = conn.execute("SELECT COUNT(*) FROM docs").fetchone()[0]
    f = conn.execute("SELECT COUNT(*) FROM docs_fts").fetchone()[0]
    e = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    return int(d), int(f), int(e)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-tables", required=True)
    ap.add_argument("--out-index", required=True)
    ap.add_argument("--with-embeddings", action="store_true", help="Build sentence-transformer embeddings")
    ap.add_argument("--embed-model", default="all-MiniLM-L6-v2")
    ap.add_argument("--reset", action="store_true", help="Delete index file before building")
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    t0 = time.time()
    if args.reset:
        reset_index(args.out_index)

    print(f"[INFO] Reading train tables: {args.train_tables}")
    train_tables = load_train_tables(args.train_tables)

    schema = schema_docs(train_tables)
    docs = RULE_DOCS + schema
    print(f"[INFO] Total docs: {len(docs)}")

    conn = connect(args.out_index)
    try:
        init_schema(conn)
        upsert_docs(conn, docs)

        if args.with_embeddings:
            print(f"[INFO] Loading embedding model: {args.embed_model}")
            build_embeddings(conn, docs, model_name=args.embed_model, batch_size=int(args.batch_size))

        d, f, e = stats(conn)
        print(f"[OK] Wrote index: {args.out_index}")
        print(f"[STATS] docs={d} docs_fts={f} embeddings={e}")
    finally:
        conn.close()

    print(f"[DONE] elapsed={time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
