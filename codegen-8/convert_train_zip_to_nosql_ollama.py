#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
convert_train_zip_to_nosql_ollama.py

Generate MongoDB query JSON for BIRD train samples using local Ollama.
Optionally uses a small SQLite RAG index (FTS + optional embeddings) built by rag_index_build.py.

Key features:
- Threaded generation with per-process request semaphore (ollama-parallel)
- SQLite prompt+response cache (ollama_cache.sqlite) to avoid re-generation
- RAG retrieval (safe FTS query escaping; no "fts5 syntax error near .")
- Lightweight pipeline sanitizer + schema-aware join fixups (helps execution accuracy)

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


# ----------------------------
# SQL sanitizer (strip trailing db_id / junk)
# ----------------------------

_SQL_END_RE = re.compile(r"(;)?\s*$", re.M)


def sanitize_sql(sql: str, db_id: str) -> str:
    if not sql:
        return sql
    s = sql.strip()

    # Common dataset artifact: "\t<db_id>" appended
    # Remove if the tail token matches db_id exactly.
    tail = s.split()[-1] if s.split() else ""
    if tail == db_id:
        s = s[: s.rfind(tail)].rstrip()

    # Also handle tab-separated db_id
    if ("\t" + db_id) in s:
        s = s.replace("\t" + db_id, "").rstrip()

    # Remove stray trailing characters after a balanced SQL statement (best-effort)
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
            # Primary keys in BIRD can be single-column (ints) or composite (lists of ints).
            # Normalize them into a hashable set of tuples, e.g. {(3,), (1,2)}.
            pk_raw = db.get("primary_keys") or []
            pk: set = set()
            for _pk in pk_raw:
                if isinstance(_pk, (list, tuple)):
                    pk.add(tuple(_pk))
                else:
                    pk.add((_pk,))
            fk = db.get("foreign_keys") or []
            # table idx -> name
            tnames = {i: table_names[i] for i in range(len(table_names))}
            # table -> set(cols)
            tcols: Dict[str, set] = {tnames[i]: set() for i in tnames}
            # col index -> (table, col)
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

            # fk edges: (t1,col1) -> (t2,col2)
            fk_edges: List[Tuple[Tuple[str,str], Tuple[str,str]]] = []
            for pair in fk:
                if isinstance(pair, (list, tuple)) and len(pair) == 2:
                    a, b = pair
                    if a in col_index_map and b in col_index_map:
                        fk_edges.append((col_index_map[a], col_index_map[b]))

            self.db_map[db_id] = {
                "tables": list(tcols.keys()),
                "tcols": {k: sorted(list(v)) for k,v in tcols.items()},
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

    def join_key(self, db_id: str, left_table: str, right_table: str) -> Optional[Tuple[str,str]]:
        """
        Return a plausible (left_field, right_field) to join left->right based on FK edges.
        """
        d = self.db_map.get(db_id) or {}
        for (t1,c1),(t2,c2) in d.get("fk_edges") or []:
            if t1==left_table and t2==right_table:
                return (c1, c2)
            if t2==left_table and t1==right_table:
                return (c2, c1)
        # fallback: common patterns
        for cand in ["id", f"{right_table}_id", f"{left_table}_id"]:
            if cand in self.columns(db_id, left_table) and cand in self.columns(db_id, right_table):
                return (cand, cand)
        return None


# ----------------------------
# RAG Index (FTS5 + optional embeddings)
# ----------------------------

def _fts_escape(query: str) -> str:
    """Make an FTS5-safe query string.

    FTS5 can throw errors for punctuation (e.g., '?', ':', '.') or for tokens that
    accidentally form boolean syntax. We take a conservative approach:
      - keep only word tokens [A-Za-z0-9_]+
      - quote each token to avoid it being parsed as an operator
      - join with AND
    """
    q = (query or "").replace("\n", " ").strip()
    if not q:
        return ""
    toks = re.findall(r"[A-Za-z0-9_]+", q)
    if not toks:
        return ""
    # Limit length to keep MATCH fast and avoid SQLite limits.
    toks = toks[:40]
    # Quote tokens so even keywords like AND/OR/NEAR are treated as literals.
    toks = [f'"{t}"' for t in toks]
    return " AND ".join(toks)


class RagIndex:
    def __init__(self, sqlite_path: str, use_embeddings: bool = True, embed_model: str = "all-MiniLM-L6-v2"):
        self.sqlite_path = sqlite_path
        self.use_embeddings = use_embeddings
        self.embed_model = embed_model
        self._model = None
        self._lock = threading.Lock()

    def _ensure_model(self):
        if not self.use_embeddings:
            return
        with self._lock:
            if self._model is None:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.embed_model)

    def retrieve(self, db_id: str, query: str, k: int = 5, candidates: int = 50) -> List[Dict[str, Any]]:
        conn = sqlite3.connect(self.sqlite_path)
        try:
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
        finally:
            conn.close()


# ----------------------------
# Ollama client + cache
# ----------------------------

class CacheDB:
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        conn = sqlite3.connect(self.path)
        try:
            conn.execute(
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
            conn.commit()
        finally:
            conn.close()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            conn = sqlite3.connect(self.path)
            try:
                row = conn.execute("SELECT response_json FROM cache WHERE key = ?", (key,)).fetchone()
                if not row:
                    return None
                return json.loads(row[0])
            finally:
                conn.close()

    def put(self, key: str, model: str, prompt: str, resp: Dict[str, Any]) -> None:
        with self._lock:
            conn = sqlite3.connect(self.path)
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO cache(key, created_ts, model, prompt_sha, response_json) VALUES (?, ?, ?, ?, ?)",
                    (key, time.time(), model, sha256_text(prompt), json.dumps(resp, ensure_ascii=False)),
                )
                conn.commit()
            finally:
                conn.close()


def ollama_generate(
    url: str,
    model: str,
    prompt: str,
    timeout_s: int,
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,
        },
    }
    r = requests.post(url.rstrip("/") + "/api/generate", json=payload, timeout=timeout_s)
    r.raise_for_status()
    return r.json()


# ----------------------------
# Pipeline sanitization + join fixups
# ----------------------------

UNSUPPORTED_EXPR_KEYS = {"$currentDate", "$date", "$dateConstant"}

def _is_stage(x: Any) -> bool:
    return isinstance(x, dict) and len(x) == 1 and next(iter(x)).startswith("$")


def sanitize_pipeline(pipeline: Any) -> Optional[List[Dict[str, Any]]]:
    if pipeline is None:
        return []
    if isinstance(pipeline, dict):
        # sometimes model returns {"pipeline":[...]}
        if "pipeline" in pipeline and isinstance(pipeline["pipeline"], list):
            pipeline = pipeline["pipeline"]
        else:
            return None
    if not isinstance(pipeline, list):
        return None

    out = []
    for st in pipeline:
        if not isinstance(st, dict):
            return None
        # Allow shorthand stage like {"field": 1} by converting to $match? too risky
        if len(st) != 1 or not next(iter(st)).startswith("$"):
            # best-effort: if keys "match"/"project"/"sort"/"limit"/"lookup" exist, map them
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
            # remove unsupported expressions
            cleaned = {}
            for k, v in body.items():
                if isinstance(v, dict) and any(x in UNSUPPORTED_EXPR_KEYS for x in v.keys()):
                    continue
                cleaned[k] = v
            body = cleaned
        out.append({op: body})
    return out


def _match_fields(match_obj: Any) -> List[str]:
    fields = []
    if not isinstance(match_obj, dict):
        return fields
    for k, v in match_obj.items():
        if k.startswith("$"):
            # $and / $or
            if isinstance(v, list):
                for sub in v:
                    fields.extend(_match_fields(sub))
        else:
            fields.append(k)
    return fields


def apply_schema_join_fixups(db_id: str, base_collection: str, pipeline: List[Dict[str, Any]], schema: SchemaStore) -> List[Dict[str, Any]]:
    """
    If $match refers to fields not in base collection but in another table, insert $lookup+$unwind.
    Also handles already-prefixed keys like "ratings.user_id" by ensuring $lookup ratings as "ratings".
    """
    if not pipeline:
        return pipeline

    base_cols = set(schema.columns(db_id, base_collection))
    if not base_cols:
        return pipeline

    # locate first $match
    match_idx = None
    for i, st in enumerate(pipeline):
        if "$match" in st:
            match_idx = i
            break
    if match_idx is None:
        return pipeline

    match_obj = pipeline[match_idx]["$match"]
    fields = _match_fields(match_obj)

    # determine required joins
    required_tables: Dict[str, Dict[str, Any]] = {}  # table -> {localField, foreignField, as}
    for f in fields:
        if "." in f:
            tbl, col = f.split(".", 1)
            if tbl != base_collection:
                required_tables[tbl] = {"as": tbl, "col": col, "prefixed": True}
            continue
        if f in base_cols:
            continue
        t = schema.table_for_field(db_id, f)
        if t and t != base_collection:
            required_tables[t] = {"as": t, "col": f, "prefixed": False}

    if not required_tables:
        return pipeline

    # build lookup stages before match
    new_stages = []
    for t, info in required_tables.items():
        jk = schema.join_key(db_id, base_collection, t)
        if not jk:
            continue
        left_field, right_field = jk
        new_stages.append({"$lookup": {"from": t, "localField": left_field, "foreignField": right_field, "as": t}})
        new_stages.append({"$unwind": {"path": f"${t}", "preserveNullAndEmptyArrays": False}})

    if not new_stages:
        return pipeline

    # rewrite match keys that are unprefixed but belong to joined tables
    def rewrite_match(obj: Any) -> Any:
        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                if k.startswith("$"):
                    if isinstance(v, list):
                        out[k] = [rewrite_match(x) for x in v]
                    else:
                        out[k] = rewrite_match(v)
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


# ----------------------------
# Prompting
# ----------------------------

PROMPT_TEMPLATE = """You are an expert at translating SQLite SQL to MongoDB aggregation pipelines.

Return ONLY a single JSON object (no markdown).
Schema:
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

Rules:
- Use ONLY MongoDB aggregation (operation must be "aggregate").
- Every pipeline stage must have EXACTLY ONE key ($match/$lookup/$unwind/$group/$project/$sort/$limit/$count).
- If SQL uses JOIN, use $lookup + $unwind, then filter on joined fields (prefixed with the joined table name).
- Do NOT use unsupported operators ($date, $currentDate, $dateConstant).
- Do NOT invent _id as join key; use schema keys.
- End with $project selecting only required output fields, set _id:0 unless SQL selects it.
"""


def build_schema_context(schema: SchemaStore, db_id: str, rag_hits: List[Dict[str, Any]]) -> str:
    # Keep it short: overview + any retrieved table docs
    chunks = []
    # rag hits
    for h in rag_hits:
        chunks.append(f"[{h.get('doc_type')}] {h.get('title')}\n{h.get('text')}")
    # fallback minimal if no hits
    if not chunks:
        tables = schema.all_tables(db_id)[:20]
        chunks.append(f"DB {db_id} tables: {', '.join(tables)}")
    return "\n\n".join(chunks)[:3500]


# ----------------------------
# Main
# ----------------------------


def load_done_idxs(*paths: str) -> set:
    done=set()
    for p in paths:
        if not p or not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line=line.strip()
                if not line:
                    continue
                try:
                    obj=json.loads(line)
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
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--ollama-model", required=True)
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--ollama-parallel", type=int, default=2)
    ap.add_argument("--timeout-s", type=int, default=120)
    ap.add_argument("--resume", action="store_true", help="Skip idx already present in output/error jsonl")
    ap.add_argument("--export-dbs", action="store_true", help="Only export sqlite dbs from zip into out-dir/train_databases")
    # RAG
    ap.add_argument("--rag-index", default=None)
    ap.add_argument("--rag-k", type=int, default=5)
    ap.add_argument("--rag-candidates", type=int, default=50)
    ap.add_argument("--rag-use-embeddings", action="store_true", help="Rerank with embeddings if present")
    ap.add_argument("--rag-embed-model", default="all-MiniLM-L6-v2")
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # export dbs if requested
    if args.export_dbs:
        out_root = os.path.join(args.out_dir, "train_databases")
        ensure_dir(out_root)
        print(f"[INFO] Exporting DBs from zip -> {out_root}")
        with zipfile.ZipFile(args.train_dbs_zip, "r") as zf:
            zf.extractall(out_root)
        print("[OK] Export complete.")
        return

    # load samples
    samples = safe_read_json(args.train_json)
    with open(args.train_gold_sql, "r", encoding="utf-8") as f:
        gold_sql_lines = [line.rstrip("\n") for line in f]

    schema = SchemaStore(args.train_tables)

    rag = None
    if args.rag_index:
        rag = RagIndex(args.rag_index, use_embeddings=bool(args.rag_use_embeddings), embed_model=args.rag_embed_model)

    # cache
    cache_path = os.path.join(args.out_dir, "ollama_cache.sqlite")
    cache = CacheDB(cache_path)

    out_path = os.path.join(args.out_dir, f"nosql_queries_{args.start}_{args.end}.jsonl")
    err_path = os.path.join(args.out_dir, f"errors_{args.start}_{args.end}.jsonl")
    done = load_done_idxs(out_path, err_path) if args.resume else set()

    semaphore = threading.Semaphore(int(args.ollama_parallel))

    def one(i: int) -> Dict[str, Any]:
        ex = samples[i]
        db_id = ex["db_id"]
        question = ex["question"]
        sql = gold_sql_lines[i] if i < len(gold_sql_lines) else ex.get("sql", "")
        sql = sanitize_sql(sql, db_id)

        rag_hits = []
        if rag is not None:
            rag_hits = rag.retrieve(db_id=db_id, query=f"{question}\n{sql}", k=int(args.rag_k), candidates=int(args.rag_candidates))

        schema_ctx = build_schema_context(schema, db_id, rag_hits)
        prompt = PROMPT_TEMPLATE.format(schema_context=schema_ctx, question=question, sql=sql)

        cache_key = sha256_text(f"{args.ollama_model}|{prompt}")
        cached = cache.get(cache_key)
        if cached:
            resp = cached
        else:
            with semaphore:
                t0 = now_s()
                resp = ollama_generate(args.ollama_url, args.ollama_model, prompt, timeout_s=int(args.timeout_s))
                resp["_latency_s"] = now_s() - t0
            cache.put(cache_key, args.ollama_model, prompt, resp)

        text = resp.get("response", "") if isinstance(resp, dict) else ""
        latency = float(resp.get("_latency_s", 0.0)) if isinstance(resp, dict) else 0.0

        record = {
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
            "latency_s": latency,
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
            # join fixups
            pipeline3 = apply_schema_join_fixups(db_id, col, pipeline2, schema)
            record["nosql"] = {"collection": col, "operation": "aggregate", "pipeline": pipeline3}
            record["ok"] = True
            record["rag_hits"] = [{"doc_id": h["doc_id"], "title": h["title"], "score": h.get("emb_score", h.get("score", 0.0))} for h in rag_hits]
        except Exception as e:
            record["error"] = str(e)

        return record

    # run
    subset = [i for i in range(int(args.start), int(args.end)) if i not in done]
    ok, bad = 0, 0
    with open(out_path, "a", encoding="utf-8") as out_f, open(err_path, "a", encoding="utf-8") as err_f:
        with cf.ThreadPoolExecutor(max_workers=int(args.workers)) as ex:
            futures = {ex.submit(one, i): i for i in subset}
            for fut in tqdm(cf.as_completed(futures), total=len(futures), desc="Converting"):
                rec = fut.result()
                if rec.get("ok"):
                    ok += 1
                    out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    out_f.flush()
                else:
                    bad += 1
                    err_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    err_f.flush()

    print(f"[OK] Wrote: {out_path} (ok={ok}, bad={bad})")
    print(f"[OK] Errors: {err_path}")


if __name__ == "__main__":
    main()
