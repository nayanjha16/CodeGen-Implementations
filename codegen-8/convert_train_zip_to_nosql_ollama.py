#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_train_zip_to_nosql_ollama.py

Generate MongoDB query JSON for BIRD train samples using local Ollama.
Optionally uses a small SQLite RAG index (FTS + optional embeddings) built by rag_index_build.py.

Key features (v2 optimizations):
- Uses OpenAI-compatible /v1/chat/completions (faster + more consistent JSON)
- Threaded generation with request semaphore (ollama-parallel)
- HTTP connection pooling via requests.Session (big win)
- SQLite cache uses a single long-lived connection (WAL) instead of open/close each call
- RAG retrieval uses thread-local SQLite connections (WAL) instead of open/close each call
- Buffered output flush (reduces fsync pressure)
- Adds UNION-ALL auto-repair heuristic for common "multi lookup + group" failure pattern

Output JSONL records:
{idx, db_id, question, sql, nosql, ok, error, traceback, latency_s, rag_hits?}
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm


# ----------------------------
# Utils
# ----------------------------

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()


def safe_read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def now_s() -> float:
    return time.time()


def clamp_text(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return s
    return s[:max_chars]


# ----------------------------
# SQL sanitizer (strip trailing db_id / junk)
# ----------------------------

_SQL_END_RE = re.compile(r"(;)?\s*$", re.M)


def sanitize_sql(sql: str, db_id: str) -> str:
    if not sql:
        return sql
    s = sql.strip()

    # Common dataset artifact: "\t<db_id>" appended
    tail = s.split()[-1] if s.split() else ""
    if tail == db_id:
        s = s[: s.rfind(tail)].rstrip()

    # Also handle tab-separated db_id
    if ("\t" + db_id) in s:
        s = s.replace("\t" + db_id, "").rstrip()

    # Remove stray trailing ; / whitespace
    s = _SQL_END_RE.sub("", s).strip()
    return s


# ----------------------------
# Schema loader for join fixups
# ----------------------------

class SchemaStore:
    """
    Loads train_tables.json and provides:
    - table -> columns
    - FK adjacency between tables (by actual column names)
    """

    def __init__(self, train_tables_path: str):
        self.raw = safe_read_json(train_tables_path)
        self.db_map: Dict[str, Dict[str, Any]] = {}
        self._build()

    def _build(self):
        for db in self.raw:
            db_id = db.get("db_id") or ""
            if not db_id:
                continue

            table_names = db.get("table_names_original") or db.get("table_names") or []
            col_names = db.get("column_names_original") or db.get("column_names") or []
            fk = db.get("foreign_keys") or []

            tnames = {i: table_names[i] for i in range(len(table_names))}
            tcols: Dict[str, set] = {tnames[i]: set() for i in tnames}

            col_index_map: Dict[int, Tuple[str, str]] = {}
            for i, col in enumerate(col_names):
                if not isinstance(col, (list, tuple)) or len(col) != 2:
                    continue
                t_idx, c_name = col
                if t_idx == -1:
                    continue
                tname = tnames.get(t_idx)
                if not tname:
                    continue
                tcols[tname].add(c_name)
                col_index_map[i] = (tname, c_name)

            fk_edges: List[Tuple[Tuple[str, str], Tuple[str, str]]] = []
            for pair in fk:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    a, b = pair
                    if a in col_index_map and b in col_index_map:
                        fk_edges.append((col_index_map[a], col_index_map[b]))

            self.db_map[db_id] = {
                "tables": list(tcols.keys()),
                "tcols": {k: sorted(list(v)) for k, v in tcols.items()},
                "fk_edges": fk_edges,
            }

    def columns(self, db_id: str, table: str) -> List[str]:
        d = self.db_map.get(db_id) or {}
        return list((d.get("tcols") or {}).get(table) or [])

    def all_tables(self, db_id: str) -> List[str]:
        d = self.db_map.get(db_id) or {}
        return list(d.get("tables") or [])

    def table_for_field(self, db_id: str, field: str) -> Optional[str]:
        d = self.db_map.get(db_id) or {}
        tcols = d.get("tcols") or {}
        for t, cols in tcols.items():
            if field in cols:
                return t
        return None

    def join_key(self, db_id: str, left_table: str, right_table: str) -> Optional[Tuple[str, str]]:
        """
        Return a plausible (left_field, right_field) to join left->right based on FK edges.
        """
        d = self.db_map.get(db_id) or {}
        for (t1, c1), (t2, c2) in d.get("fk_edges") or []:
            if t1 == left_table and t2 == right_table:
                return (c1, c2)
            if t2 == left_table and t1 == right_table:
                return (c2, c1)

        # fallback: try common patterns if schemas are simple
        common = set(self.columns(db_id, left_table)).intersection(self.columns(db_id, right_table))
        for cand in ["id", f"{right_table}_id", f"{left_table}_id"]:
            if cand in common:
                return (cand, cand)
        return None


# ----------------------------
# RAG Index (FTS5 + optional embeddings)
# ----------------------------

def _fts_escape(query: str) -> str:
    """
    Make an FTS5-safe query string:
      - keep only word tokens [A-Za-z0-9_]+
      - quote each token
      - join with AND
    """
    q = (query or "").replace("\n", " ").strip()
    if not q:
        return ""
    toks = re.findall(r"[A-Za-z0-9_]+", q)
    if not toks:
        return ""
    toks = toks[:40]
    toks = [f'"{t}"' for t in toks]
    return " AND ".join(toks)


class RagIndex:
    def __init__(self, sqlite_path: str, use_embeddings: bool = True, embed_model: str = "all-MiniLM-L6-v2"):
        self.sqlite_path = sqlite_path
        self.use_embeddings = use_embeddings
        self.embed_model = embed_model
        self._model = None
        self._model_lock = threading.Lock()
        self._tls = threading.local()

    def _get_conn(self) -> sqlite3.Connection:
        conn = getattr(self._tls, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
            # speed-friendly pragmas (safe enough for read-only retrieval)
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                conn.execute("PRAGMA temp_store=MEMORY;")
                conn.execute("PRAGMA cache_size=-200000;")  # ~200MB if possible
            except Exception:
                pass
            self._tls.conn = conn
        return conn

    def _ensure_model(self):
        if not self.use_embeddings:
            return
        with self._model_lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.embed_model)

    def retrieve(self, db_id: str, query: str, k: int = 5, candidates: int = 50) -> List[Dict[str, Any]]:
        conn = self._get_conn()
        q = _fts_escape(query)
        if not q:
            return []

        rows = conn.execute(
            """
            SELECT d.doc_id, d.db_id, d.doc_type, d.title, d.text, bm25(docs_fts) AS score
            FROM docs_fts
            JOIN docs d ON d.doc_id = docs_fts.doc_id
            WHERE docs_fts MATCH ? AND (d.db_id = ? OR d.db_id='*')
            ORDER BY score
            LIMIT ?
            """,
            (q, db_id, int(candidates)),
        ).fetchall()

        cands = [
            {"doc_id": r[0], "db_id": r[1], "doc_type": r[2], "title": r[3], "text": r[4], "score": float(-r[5])}
            for r in rows
        ]

        if not self.use_embeddings:
            return cands[:k]

        # rerank
        self._ensure_model()
        try:
            import numpy as np
        except Exception:
            return cands[:k]

        doc_ids = [c["doc_id"] for c in cands]
        if not doc_ids:
            return []

        qmarks = ",".join(["?"] * len(doc_ids))
        emb_rows = conn.execute(
            f"SELECT doc_id, dim, vec FROM embeddings WHERE doc_id IN ({qmarks})",
            doc_ids,
        ).fetchall()
        emb_map = {r[0]: (int(r[1]), r[2]) for r in emb_rows}
        if not emb_map:
            return cands[:k]

        qv = self._model.encode([query], normalize_embeddings=True)[0]
        qv = np.asarray(qv, dtype=np.float32)

        rescored = []
        for c in cands:
            doc_id = c["doc_id"]
            if doc_id not in emb_map:
                continue
            dim, vec = emb_map[doc_id]
            dv = np.frombuffer(vec, dtype=np.float32, count=dim)
            cc = dict(c)
            cc["emb_score"] = float(np.dot(qv, dv))
            rescored.append(cc)

        rescored.sort(key=lambda x: x.get("emb_score", -1e9), reverse=True)
        return rescored[:k]


# ----------------------------
# HTTP session (pooling + retries)
# ----------------------------

def make_session(pool_size: int) -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=2,
        backoff_factor=0.2,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=frozenset(["POST"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(pool_connections=pool_size, pool_maxsize=pool_size, max_retries=retries)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


# ----------------------------
# Ollama client + cache (fast, single connection)
# ----------------------------

class CacheDB:
    """
    Thread-safe SQLite cache using a single long-lived connection (WAL).
    This avoids reconnect overhead per request (huge speed win).
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        try:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("PRAGMA temp_store=MEMORY;")
        except Exception:
            pass

        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache(
                key TEXT PRIMARY KEY,
                created_ts REAL NOT NULL,
                model TEXT NOT NULL,
                prompt_sha TEXT NOT NULL,
                response_json TEXT NOT NULL
            )
            """
        )
        self._conn.commit()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            row = self._conn.execute("SELECT response_json FROM cache WHERE key = ?", (key,)).fetchone()
            if not row:
                return None
            try:
                return json.loads(row[0])
            except Exception:
                return None

    def put(self, key: str, model: str, prompt: str, resp: Dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache(key, created_ts, model, prompt_sha, response_json) VALUES (?, ?, ?, ?, ?)",
                (key, time.time(), model, sha256_text(prompt), json.dumps(resp, ensure_ascii=False)),
            )
            self._conn.commit()


def ollama_chat_completions(
    session: requests.Session,
    url: str,
    model: str,
    system: str,
    user: str,
    timeout_s: int,
) -> Dict[str, Any]:
    """
    OpenAI-compatible chat endpoint:
      POST /v1/chat/completions
    """
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.0,
        "stream": False,
    }
    r = session.post(url.rstrip("/") + "/v1/chat/completions", json=payload, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


def extract_chat_text(resp: Dict[str, Any]) -> str:
    """
    Pull assistant content from chat response.
    """
    try:
        return resp["choices"][0]["message"]["content"] or ""
    except Exception:
        return ""


# ----------------------------
# Pipeline sanitization + join fixups + UNION repair
# ----------------------------

UNSUPPORTED_EXPR_KEYS = {"$currentDate", "$date", "$dateConstant"}


def sanitize_pipeline(pipeline: Any) -> Optional[List[Dict[str, Any]]]:
    if pipeline is None:
        return []
    if isinstance(pipeline, dict):
        if "pipeline" in pipeline and isinstance(pipeline["pipeline"], list):
            pipeline = pipeline["pipeline"]
        else:
            return None
    if not isinstance(pipeline, list):
        return None

    out: List[Dict[str, Any]] = []
    for st in pipeline:
        if not isinstance(st, dict):
            return None
        if len(st) != 1 or not next(iter(st)).startswith("$"):
            # small best-effort mapping
            if set(st.keys()) == {"filter"}:
                out.append({"$match": st["filter"]})
                continue
            if set(st.keys()) == {"project"}:
                out.append({"$project": st["project"]})
                continue
            if set(st.keys()) == {"sort"}:
                out.append({"$sort": st["sort"]})
                continue
            if set(st.keys()) == {"limit"}:
                out.append({"$limit": st["limit"]})
                continue
            if set(st.keys()) == {"lookup"}:
                out.append({"$lookup": st["lookup"]})
                continue
            if set(st.keys()) == {"match"}:
                out.append({"$match": st["match"]})
                continue
            if set(st.keys()) == {"group"}:
                out.append({"$group": st["group"]})
                continue
            return None

        op, body = next(iter(st.items()))
        if op == "$sort" and (not isinstance(body, dict) or len(body) == 0):
            return None
        if op == "$group" and (not isinstance(body, dict) or "_id" not in body):
            return None
        if op == "$project" and isinstance(body, dict):
            cleaned = {}
            for k, v in body.items():
                if isinstance(v, dict) and any(x in UNSUPPORTED_EXPR_KEYS for x in v.keys()):
                    continue
                cleaned[k] = v
            body = cleaned
        out.append({op: body})
    return out


def _match_fields(match_obj: Any) -> List[str]:
    fields: List[str] = []
    if not isinstance(match_obj, dict):
        return fields
    for k, v in match_obj.items():
        if k.startswith("$"):
            if isinstance(v, list):
                for sub in v:
                    fields.extend(_match_fields(sub))
        else:
            fields.append(k)
    return fields


def apply_schema_join_fixups(db_id: str, base_collection: str, pipeline: List[Dict[str, Any]], schema: SchemaStore) -> List[Dict[str, Any]]:
    """
    If $match refers to fields not in base collection but in another table, insert $lookup+$unwind
    right before the first $match.
    """
    if not pipeline:
        return pipeline

    base_cols = set(schema.columns(db_id, base_collection))
    if not base_cols:
        return pipeline

    match_idx = None
    for i, st in enumerate(pipeline):
        if "$match" in st:
            match_idx = i
            break
    if match_idx is None:
        return pipeline

    match_obj = pipeline[match_idx]["$match"]
    fields = _match_fields(match_obj)

    required_tables: Dict[str, Dict[str, Any]] = {}
    for f in fields:
        if "." in f:
            tbl, _col = f.split(".", 1)
            if tbl != base_collection:
                required_tables[tbl] = {"as": tbl, "prefixed": True}
            continue
        if f in base_cols:
            continue
        t = schema.table_for_field(db_id, f)
        if t and t != base_collection:
            required_tables[t] = {"as": t, "prefixed": False}

    if not required_tables:
        return pipeline

    new_stages: List[Dict[str, Any]] = []
    for t in required_tables.keys():
        jk = schema.join_key(db_id, base_collection, t)
        if not jk:
            continue
        left_field, right_field = jk
        new_stages.append({"$lookup": {"from": t, "localField": left_field, "foreignField": right_field, "as": t}})
        new_stages.append({"$unwind": {"path": f"${t}", "preserveNullAndEmptyArrays": False}})

    if not new_stages:
        return pipeline

    def rewrite_match(obj: Any) -> Any:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k.startswith("$"):
                    out[k] = [rewrite_match(x) for x in v] if isinstance(v, list) else rewrite_match(v)
                    continue
                if "." in k:
                    out[k] = rewrite_match(v)
                    continue
                if k in base_cols:
                    out[k] = rewrite_match(v)
                    continue
                t = schema.table_for_field(db_id, k)
                if t and t in required_tables:
                    out[f"{t}.{k}"] = rewrite_match(v)
                else:
                    out[k] = rewrite_match(v)
            return out
        if isinstance(obj, list):
            return [rewrite_match(x) for x in obj]
        return obj

    pipeline2 = list(pipeline)
    pipeline2[match_idx] = {"$match": rewrite_match(match_obj)}
    pipeline2 = pipeline2[:match_idx] + new_stages + pipeline2[match_idx:]
    return pipeline2


def _looks_like_union_all_sales_mistake(base_collection: str, pipeline: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Detect the common wrong pattern:
      product (base) -> lookup fact -> unwind -> group (loses Product_ID)
                     -> lookup another fact -> unwind -> group ...

    If detected, return a dict describing:
      {"dimension": "product", "facts": [...], "join_key": "Product_ID", "value_field": "Sales", "name_field": "Product_Name"}
    else None.
    """
    if not pipeline or base_collection is None:
        return None

    # gather lookups
    lookups: List[Dict[str, Any]] = []
    for st in pipeline:
        if "$lookup" in st and isinstance(st["$lookup"], dict):
            lk = st["$lookup"]
            if all(k in lk for k in ("from", "localField", "foreignField", "as")):
                lookups.append(lk)

    if len(lookups) < 2:
        return None

    # must be joining by a consistent key (typical: Product_ID)
    local_fields = {lk.get("localField") for lk in lookups}
    foreign_fields = {lk.get("foreignField") for lk in lookups}

    if len(local_fields) != 1 or len(foreign_fields) != 1:
        return None

    join_key = next(iter(local_fields))
    if not join_key or join_key != next(iter(foreign_fields)):
        # we only auto-repair when it’s clearly the same join key on both sides
        return None

    # check if pipeline contains group by Product_Name (or similar) BEFORE all lookups finish
    group_positions = [i for i, st in enumerate(pipeline) if "$group" in st]
    if len(group_positions) < 2:
        return None

    first_group = group_positions[0]
    # if there is another lookup after first group, it's the broken sequential aggregation pattern
    for j in range(first_group + 1, len(pipeline)):
        if "$lookup" in pipeline[j]:
            # also verify the group _id is name-like from dimension table
            gid = pipeline[first_group].get("$group", {}).get("_id")
            if isinstance(gid, str) and "Name" in gid:
                facts = [lk["from"] for lk in lookups]
                return {
                    "dimension": base_collection,
                    "facts": facts,
                    "join_key": join_key,
                    "value_field": "Sales",
                    "name_field": "Product_Name",
                }

    return None


def repair_union_all_sales_pipeline(base_collection: str, pipeline: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Rewrite the broken pattern into canonical:
      fact1
        $project {Product_ID, Sales}
        $unionWith fact2...
        $group by Product_ID sum Sales
        $lookup product
        $project {Product_Name, Total_Sales}
        $sort, $limit
    Returns {"collection": <new_collection>, "pipeline": <new_pipeline>} or None.
    """
    info = _looks_like_union_all_sales_mistake(base_collection, pipeline)
    if not info:
        return None

    dim = info["dimension"]
    facts = info["facts"]
    join_key = info["join_key"]
    value_field = info["value_field"]
    name_field = info["name_field"]

    if not facts:
        return None

    first_fact = facts[0]
    other_facts = facts[1:]

    new_pipeline: List[Dict[str, Any]] = [
        {"$project": {join_key: 1, value_field: 1}},
    ]

    for f in other_facts:
        new_pipeline.append({
            "$unionWith": {
                "coll": f,
                "pipeline": [
                    {"$project": {join_key: 1, value_field: 1}},
                ],
            }
        })

    new_pipeline += [
        {"$group": {"_id": f"${join_key}", "Total_Sales": {"$sum": f"${value_field}"}}},
        {"$lookup": {"from": dim, "localField": "_id", "foreignField": join_key, "as": dim}},
        {"$unwind": {"path": f"${dim}", "preserveNullAndEmptyArrays": False}},
        {"$project": {"_id": 0, name_field: f"${dim}.{name_field}", "Total_Sales": 1}},
    ]

    # carry over sort/limit if present; else default to top 5
    has_sort = any("$sort" in st for st in pipeline)
    has_limit = any("$limit" in st for st in pipeline)

    if has_sort:
        # use the last sort stage from original
        for st in reversed(pipeline):
            if "$sort" in st:
                new_pipeline.append(st)
                break
    else:
        new_pipeline.append({"$sort": {"Total_Sales": -1}})

    if has_limit:
        for st in reversed(pipeline):
            if "$limit" in st:
                new_pipeline.append(st)
                break
    else:
        new_pipeline.append({"$limit": 5})

    return {"collection": first_fact, "pipeline": new_pipeline}


# ----------------------------
# Prompting
# ----------------------------

SYSTEM_PROMPT = (
    "You translate SQLite SQL into MongoDB aggregation pipelines. "
    "Return ONLY a single JSON object (no markdown, no extra text). "
    "The JSON MUST parse."
)

USER_TEMPLATE = """Schema:
{schema_context}

Task:
Question: {question}
SQL: {sql}

Output JSON format (strict):
{{
  "collection": "<base_collection>",
  "operation": "aggregate",
  "pipeline": [ ... valid MongoDB aggregation stages ... ]
}}

Rules (MUST follow):
- Use ONLY aggregation (operation must be "aggregate").
- Every pipeline stage must have EXACTLY ONE key ($match/$lookup/$unwind/$group/$project/$sort/$limit/$count/$unionWith).
- If SQL uses UNION ALL across multiple tables with the SAME columns, use $unionWith to concatenate tables first, then do a SINGLE $group at the end.
- Do NOT $group and then do another $lookup to sum more facts (that loses join keys).
- If SQL uses JOIN, use $lookup + $unwind, then filter on joined fields (prefixed with joined table name).
- Do NOT invent join keys; use schema keys.
- End with $project selecting only required output fields, set _id:0 unless SQL selects it.
"""


def build_schema_context(schema: SchemaStore, db_id: str, rag_hits: List[Dict[str, Any]], max_chars: int) -> str:
    chunks: List[str] = []
    for h in rag_hits:
        chunks.append(f"[{h.get('doc_type')}] {h.get('title')}\n{h.get('text')}")
    if not chunks:
        tables = schema.all_tables(db_id)[:25]
        chunks.append(f"DB {db_id} tables: {', '.join(tables)}")
    return clamp_text("\n\n".join(chunks), max_chars)


# ----------------------------
# Main
# ----------------------------

def load_done_idxs(*paths: str) -> set:
    done = set()
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    if "idx" in obj:
                        done.add(int(obj["idx"]))
                except Exception:
                    continue
    return done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--train-tables", required=True)
    ap.add_argument("--train-gold-sql", required=True)
    ap.add_argument("--train-dbs-zip", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=100)

    # perf controls
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--ollama-parallel", type=int, default=6)
    ap.add_argument("--timeout-s", type=int, default=120)
    ap.add_argument("--flush-every", type=int, default=50)
    ap.add_argument("--schema-max-chars", type=int, default=3000)
    ap.add_argument("--prompt-max-chars", type=int, default=0, help="If >0, hard cap total user prompt size")

    # model
    ap.add_argument("--ollama-model", required=True)
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")

    # resume/export
    ap.add_argument("--resume", action="store_true", help="Skip idx already present in output/error jsonl")
    ap.add_argument("--export-dbs", action="store_true", help="Only export sqlite dbs from zip into out-dir/train_databases")

    # RAG
    ap.add_argument("--rag-index", default=None)
    ap.add_argument("--rag-k", type=int, default=5)
    ap.add_argument("--rag-candidates", type=int, default=50)
    ap.add_argument("--rag-use-embeddings", action="store_true", help="Rerank with embeddings if present")
    ap.add_argument("--rag-embed-model", default="all-MiniLM-L6-v2")

    # correctness
    ap.add_argument("--enable-union-repair", action="store_true", help="Auto-repair common UNION ALL sales pattern mistakes")

    args = ap.parse_args()

    ensure_dir(args.out_dir)

    if args.export_dbs:
        out_root = os.path.join(args.out_dir, "train_databases")
        ensure_dir(out_root)
        print(f"[INFO] Exporting DBs from zip -> {out_root}")
        with zipfile.ZipFile(args.train_dbs_zip, "r") as zf:
            zf.extractall(out_root)
        print("[OK] Export complete.")
        return

    samples = safe_read_json(args.train_json)
    with open(args.train_gold_sql, "r", encoding="utf-8") as f:
        gold_sql_lines = [line.rstrip("\n") for line in f]

    schema = SchemaStore(args.train_tables)

    rag = None
    if args.rag_index:
        rag = RagIndex(
            args.rag_index,
            use_embeddings=bool(args.rag_use_embeddings),
            embed_model=args.rag_embed_model,
        )

    cache_path = os.path.join(args.out_dir, "ollama_cache.sqlite")
    cache = CacheDB(cache_path)

    out_path = os.path.join(args.out_dir, f"nosql_queries_{args.start}_{args.end}.jsonl")
    err_path = os.path.join(args.out_dir, f"errors_{args.start}_{args.end}.jsonl")
    done = load_done_idxs(out_path, err_path) if args.resume else set()

    subset = [i for i in range(int(args.start), int(args.end)) if i not in done]
    if not subset:
        print("[INFO] Nothing to do (all done).")
        return

    # one session shared across threads (requests.Session is generally OK for concurrent POST)
    pool_size = max(int(args.workers), int(args.ollama_parallel), 8)
    session = make_session(pool_size=pool_size)

    semaphore = threading.Semaphore(int(args.ollama_parallel))

    def one(i: int) -> Dict[str, Any]:
        ex = samples[i]
        db_id = ex["db_id"]
        question = ex["question"]

        sql = gold_sql_lines[i] if i < len(gold_sql_lines) else ex.get("sql", "")
        sql = sanitize_sql(sql, db_id)

        rag_hits: List[Dict[str, Any]] = []
        if rag is not None:
            rag_hits = rag.retrieve(
                db_id=db_id,
                query=f"{question}\n{sql}",
                k=int(args.rag_k),
                candidates=int(args.rag_candidates),
            )

        schema_ctx = build_schema_context(schema, db_id, rag_hits, max_chars=int(args.schema_max_chars))
        user_prompt = USER_TEMPLATE.format(schema_context=schema_ctx, question=question, sql=sql)
        if int(args.prompt_max_chars) > 0:
            user_prompt = clamp_text(user_prompt, int(args.prompt_max_chars))

        cache_key = sha256_text(f"{args.ollama_model}|{SYSTEM_PROMPT}|{user_prompt}")
        cached = cache.get(cache_key)

        if cached:
            resp = cached
            latency = float(resp.get("_latency_s", 0.0))
            text = resp.get("_text", "") or ""
        else:
            with semaphore:
                t0 = now_s()
                raw = ollama_chat_completions(
                    session=session,
                    url=args.ollama_url,
                    model=args.ollama_model,
                    system=SYSTEM_PROMPT,
                    user=user_prompt,
                    timeout_s=int(args.timeout_s),
                )
                latency = now_s() - t0
            text = extract_chat_text(raw)
            resp = {"_latency_s": latency, "_text": text}
            cache.put(cache_key, args.ollama_model, user_prompt, resp)

        record: Dict[str, Any] = {
            "idx": i,
            "question_id": ex.get("question_id"),
            "db_id": db_id,
            "question": question,
            "sql": sql,
            "nosql": None,
            "raw": text,
            "ok": False,
            "error": None,
            "traceback": None,
            "latency_s": float(latency),
        }

        try:
            obj = json.loads(text)
            col = obj.get("collection")
            op = obj.get("operation")
            pipeline = obj.get("pipeline")

            if op != "aggregate":
                raise ValueError("operation must be 'aggregate'")

            pipeline2 = sanitize_pipeline(pipeline)
            if pipeline2 is None:
                raise ValueError("Invalid pipeline structure after sanitize")

            # optional UNION repair (fixes your exact failure mode)
            if args.enable_union_repair and isinstance(col, str) and col:
                repaired = repair_union_all_sales_pipeline(col, pipeline2)
                if repaired:
                    col = repaired["collection"]
                    pipeline2 = repaired["pipeline"]

            # join fixups (only makes sense if base collection is truly the starting collection)
            if isinstance(col, str) and col:
                pipeline3 = apply_schema_join_fixups(db_id, col, pipeline2, schema)
            else:
                pipeline3 = pipeline2

            record["nosql"] = {"collection": col, "operation": "aggregate", "pipeline": pipeline3}
            record["ok"] = True
            record["rag_hits"] = [
                {"doc_id": h.get("doc_id"), "title": h.get("title"), "score": h.get("emb_score", h.get("score", 0.0))}
                for h in rag_hits
            ]
        except Exception as e:
            record["error"] = str(e)

        return record

    ok, bad = 0, 0
    flush_every = max(int(args.flush_every), 1)
    out_buf: List[str] = []
    err_buf: List[str] = []

    with open(out_path, "a", encoding="utf-8") as out_f, open(err_path, "a", encoding="utf-8") as err_f:
        with cf.ThreadPoolExecutor(max_workers=int(args.workers)) as exr:
            futures = {exr.submit(one, i): i for i in subset}
            for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="Converting"):
                rec = fut.result()
                line = json.dumps(rec, ensure_ascii=False) + "\n"
                if rec.get("ok"):
                    ok += 1
                    out_buf.append(line)
                else:
                    bad += 1
                    err_buf.append(line)

                if (len(out_buf) + len(err_buf)) >= flush_every:
                    if out_buf:
                        out_f.writelines(out_buf)
                        out_buf.clear()
                    if err_buf:
                        err_f.writelines(err_buf)
                        err_buf.clear()
                    out_f.flush()
                    err_f.flush()

        # final flush
        if out_buf:
            out_f.writelines(out_buf)
            out_f.flush()
        if err_buf:
            err_f.writelines(err_buf)
            err_f.flush()

    print(f"[OK] Wrote: {out_path} (ok={ok}, bad={bad})")
    print(f"[OK] Errors: {err_path}")


if __name__ == "__main__":
    main()

