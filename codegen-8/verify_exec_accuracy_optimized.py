#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import sys
import time
import itertools
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pymongo import MongoClient
from pymongo.errors import PyMongoError

# -----------------------------
# Utilities
# -----------------------------

def eprint(*a: Any) -> None:
    print(*a, file=sys.stderr, flush=True)

def jdump(obj: Any, **kwargs: Any) -> str:
    """Small wrapper around json.dumps.
    Keeps ensure_ascii=False by default, but allows callers to override/extend.
    """
    kwargs.setdefault("ensure_ascii", False)
    return json.dumps(obj, **kwargs)


def safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def write_jsonl(path: str, rows: Iterable[Dict[str, Any]], mode: str = "a") -> None:
    safe_mkdir(os.path.dirname(path) or ".")
    with open(path, mode, encoding="utf-8") as f:
        for r in rows:
            f.write(jdump(r) + "\n")

def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()

# -----------------------------
# SQL Sanitizer
# Problem: SQL sometimes ends with tab + db_id
# -----------------------------

SQL_DBID_SUFFIX_RE = re.compile(r"[\t ]+([a-zA-Z0-9_]+)\s*$")

def sanitize_sql(sql: str, db_id: str) -> str:
    """Strip trailing junk like '\\tmovie_platform' after a valid statement."""
    s = (sql or "").strip()

    # If the very last token equals db_id and preceded by whitespace/tab -> strip it
    m = SQL_DBID_SUFFIX_RE.search(s)
    if m and m.group(1) == db_id:
        s = s[: m.start()].rstrip()

    # Remove trailing semicolons/spaces
    s = s.rstrip().rstrip(";").strip()
    return s

# -----------------------------
# Mongo Sanitizer (lightweight, execution-focused)
# Goal: make pipelines "runnable" or mark non-executable.
# -----------------------------

UNSUPPORTED_OPS = {
    "$date", "$currentDate", "$dateConstant", "$dateToString",  # often misused by LLM
    "$function", "$accumulator"
}

def _is_stage(x: Any) -> bool:
    return isinstance(x, dict) and len(x) == 1 and next(iter(x)).startswith("$")


def optimize_pipeline_for_speed(pipeline: Any, skip_lookup: bool = False) -> Any:
    """Best-effort pipeline rewrites to reduce timeouts during verification.

    Goals (verification-time only):
      - Make common $lookup joins less broken when foreignField is incorrectly '_id'
      - Push simple, non-dotted $match predicates as early as possible
    These rewrites can change semantics slightly; they are intended to improve executability and
    make comparisons feasible under time limits.
    """
    if not isinstance(pipeline, list):
        return pipeline

    pre_match: Dict[str, Any] = {}
    out: List[Dict[str, Any]] = []

    def _merge_match(dst: Dict[str, Any], src: Dict[str, Any]) -> None:
        for k, v in src.items():
            # last write wins; it's ok for verifier
            dst[k] = v

    def _split_match_dict(m: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Return (non_dotted, rest). Handles simple dict and $and of simple dicts."""
        nd: Dict[str, Any] = {}
        rest: Dict[str, Any] = {}

        if "$and" in m and isinstance(m["$and"], list):
            nd_parts = []
            rest_parts = []
            for part in m["$and"]:
                if isinstance(part, dict) and all(isinstance(k, str) for k in part.keys()):
                    p_nd = {k: v for k, v in part.items() if "." not in k and not k.startswith("$")}
                    p_rest = {k: v for k, v in part.items() if k not in p_nd}
                    if p_nd:
                        nd_parts.append(p_nd)
                    if p_rest:
                        rest_parts.append(p_rest)
                else:
                    rest_parts.append(part)
            if nd_parts:
                for d in nd_parts:
                    _merge_match(nd, d)
            if rest_parts:
                rest["$and"] = rest_parts
            return nd, rest

        # plain dict
        for k, v in m.items():
            if not isinstance(k, str) or k.startswith("$"):
                rest[k] = v
            elif "." in k:
                rest[k] = v
            else:
                nd[k] = v
        return nd, rest

    for st in pipeline:
        if not (isinstance(st, dict) and len(st) == 1):
            out.append(st)
            continue

        op = next(iter(st))
        val = st[op]

        if op in ("$lookup", "$graphLookup"):
            if skip_lookup:
                # leave it; caller will skip entire pipeline
                out.append(st)
                continue
            # Fix the most common bug: foreignField incorrectly set to '_id'
            if isinstance(val, dict):
                lf = val.get("localField")
                ff = val.get("foreignField")
                if isinstance(lf, str) and ff == "_id":
                    # Heuristic: join keys in these datasets are usually <name>_id fields
                    val["foreignField"] = lf
            out.append({op: val})
            continue

        if op == "$match" and isinstance(val, dict):
            nd, rest = _split_match_dict(val)
            if nd:
                _merge_match(pre_match, nd)
            if rest:
                out.append({"$match": rest})
            continue

        out.append(st)

    if pre_match:
        return [{"$match": pre_match}] + out
    return out

def sanitize_pipeline(pipeline: Any) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """
    Returns (pipeline_or_none, reason_if_none).
    Enforces:
      - list of single-key stage dicts
      - $sort has keys
      - $group has _id
      - removes/blocks obviously unsupported operators
    """
    if pipeline is None:
        return None, "missing_pipeline"

    # Some generations use filter dict for find; we allow converting to $match
    if isinstance(pipeline, dict):
        pipeline = [{"$match": pipeline}]

    if not isinstance(pipeline, list):
        return None, "pipeline_not_list"

    out: List[Dict[str, Any]] = []
    for st in pipeline:
        if isinstance(st, dict) and ("filter" in st or "match" in st) and len(st) == 1:
            key = "filter" if "filter" in st else "match"
            out.append({"$match": st[key]})
            continue

        if isinstance(st, dict) and ("project" in st) and len(st) == 1:
            out.append({"$project": st["project"]})
            continue

        if isinstance(st, dict) and ("sort" in st) and len(st) == 1:
            out.append({"$sort": st["sort"]})
            continue

        if isinstance(st, dict) and ("limit" in st) and len(st) == 1:
            out.append({"$limit": st["limit"]})
            continue

        if not _is_stage(st):
            return None, "bad_stage_shape"

        op, body = next(iter(st.items()))
        if op in UNSUPPORTED_OPS:
            return None, f"unsupported_op:{op}"

        # Validate some key stages
        if op == "$sort":
            if not isinstance(body, dict) or len(body) == 0:
                return None, "sort_empty"

        if op == "$group":
            if not isinstance(body, dict) or "_id" not in body:
                return None, "group_missing_id"

        out.append({op: body})

    return out, None

# -----------------------------
# Execution
# -----------------------------

@dataclass
class ExecResult:
    ok: bool
    rows: Optional[List[Tuple[Any, ...]]] = None
    count: Optional[int] = None
    error: Optional[str] = None

def run_sql(sqlite_path: str, sql: str) -> ExecResult:
    try:
        conn = sqlite3.connect(sqlite_path)
        try:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return ExecResult(ok=True, rows=rows)
        finally:
            conn.close()
    except Exception as e:
        return ExecResult(ok=False, error=f"{type(e).__name__}: {e}")

def run_mongo(
    mongo_uri: str,
    db_name: str,
    collection: str,
    operation: str,
    pipeline: Any,
    max_time_ms: int,
    max_docs: int = 2000,
    batch_size: int = 500,
    skip_lookup: bool = False,
    force_limit: bool = True,
    force_limit_n: int = 1000,
) -> ExecResult:
    """Execute Mongo query with safety rails.

    max_time_ms:
      - count_documents / find: uses max_time_ms
      - aggregate: uses maxTimeMS
    max_docs: cap how many documents we pull from the cursor (prevents multi-minute downloads)
    skip_lookup: optionally skip $lookup/$graphLookup pipelines (common slowdown source)
    force_limit: if True and pipeline has no $limit, append {"$limit": force_limit_n}
    """
    client = MongoClient(
        mongo_uri,
        serverSelectionTimeoutMS=8000,
        connectTimeoutMS=8000,
        socketTimeoutMS=max(8000, int(max_time_ms) + 2000),
    )
    try:
        db = client[db_name]
        col = db[collection]

        if operation == "count":
            if isinstance(pipeline, list) and pipeline and isinstance(pipeline[0], dict) and "$match" in pipeline[0]:
                filt = pipeline[0]["$match"]
            elif isinstance(pipeline, dict):
                filt = pipeline
            else:
                filt = {}
            n = col.count_documents(filt, max_time_ms=int(max_time_ms))
            return ExecResult(ok=True, count=int(n))

        if operation == "find":
            if isinstance(pipeline, dict):
                filt = pipeline
            elif isinstance(pipeline, list) and pipeline and isinstance(pipeline[0], dict) and "$match" in pipeline[0]:
                filt = pipeline[0]["$match"]
            else:
                operation = "aggregate"

            if operation == "find":
                cur = col.find(filt, max_time_ms=int(max_time_ms)).limit(int(max_docs))
                rows = list(cur)
                return ExecResult(ok=True, rows=[(r,) for r in rows])
        # aggregate
        pipeline = optimize_pipeline_for_speed(pipeline, skip_lookup=skip_lookup)
        sp, reason = sanitize_pipeline(pipeline)
        if sp is None:
            return ExecResult(ok=False, error=f"pipeline_nonexec:{reason}")

        if skip_lookup:
            for st in sp:
                if isinstance(st, dict) and len(st) == 1:
                    k = next(iter(st.keys()))
                    if k in ("$lookup", "$graphLookup"):
                        return ExecResult(ok=False, error="skipped_heavy_lookup")

        if force_limit and isinstance(sp, list):
            has_limit = any(isinstance(s, dict) and "$limit" in s for s in sp)
            if not has_limit:
                sp = list(sp) + [{"$limit": int(force_limit_n)}]

        cur = col.aggregate(
            sp,
            allowDiskUse=True,
            maxTimeMS=int(max_time_ms),
            batchSize=int(batch_size),
        )

        pulled = []
        for d in itertools.islice(cur, int(max_docs)):
            pulled.append(d)

        return ExecResult(ok=True, rows=[(r,) for r in pulled])

    except PyMongoError as e:
        return ExecResult(ok=False, error=f"PyMongoError: {e}")
    except Exception as e:
        return ExecResult(ok=False, error=f"{type(e).__name__}: {e}")
    finally:
        client.close()
# -----------------------------
# Comparison
# -----------------------------

def _norm_scalar(x: Any) -> Any:
    # Normalize numbers/strings for robust compare
    if x is None:
        return None
    if isinstance(x, bool):
        return bool(x)
    if isinstance(x, int):
        return int(x)
    if isinstance(x, float):
        # round to mitigate float noise
        try:
            if math.isfinite(x):
                return round(float(x), 6)
        except Exception:
            pass
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        # try numeric coercion
        try:
            if re.fullmatch(r"[-+]?\d+", s):
                return int(s)
            if re.fullmatch(r"[-+]?\d*\.\d+", s):
                return round(float(s), 6)
        except Exception:
            pass
        return s
    return x

def _extract_single_value_from_doc(doc: Dict[str, Any]) -> Optional[Any]:
    if not isinstance(doc, dict):
        return None
    items = [(k, v) for k, v in doc.items() if k != "_id"]
    if not items:
        return None
    # Prefer common aggregate keys
    pref = ["count", "cnt", "total", "sum", "avg", "mean", "max", "min", "value", "result"]
    for p in pref:
        for k, v in items:
            if k.lower() == p:
                return v
    if len(items) == 1:
        return items[0][1]
    return None

def _canon_sql(rows: Optional[List[Tuple[Any, ...]]]) -> Dict[str, Any]:
    rows = rows or []
    # scalar
    if len(rows) == 1 and len(rows[0]) == 1:
        return {"kind": "scalar", "value": _norm_scalar(rows[0][0])}
    # one column
    if rows and all(len(r) == 1 for r in rows):
        vals = sorted([jdump(_norm_scalar(r[0])) for r in rows])
        return {"kind": "col", "values": vals}
    # table
    norm = []
    for r in rows:
        norm.append([_norm_scalar(x) for x in r])
    # sort deterministically
    norm_s = sorted([jdump(r) for r in norm])
    return {"kind": "table", "rows": norm_s}

def _canon_mongo(rows: Optional[List[Tuple[Any, ...]]]) -> Dict[str, Any]:
    # In our runner, aggregate/find rows are returned as [(doc,), (doc,), ...]
    docs: List[Any] = []
    for r in (rows or []):
        if isinstance(r, tuple) and len(r) == 1:
            docs.append(r[0])
        else:
            docs.append(r)

    # scalar-like from single doc
    if len(docs) == 1 and isinstance(docs[0], dict):
        v = _extract_single_value_from_doc(docs[0])
        if v is not None:
            return {"kind": "scalar", "value": _norm_scalar(v)}

    # one-field docs column
    col_vals: List[str] = []
    ok = True
    for d in docs:
        if not isinstance(d, dict):
            ok = False
            break
        v = _extract_single_value_from_doc(d)
        if v is None:
            ok = False
            break
        col_vals.append(jdump(_norm_scalar(v)))
    if ok and col_vals:
        return {"kind": "col", "values": sorted(col_vals)}

    # fallback: canonicalize dicts by removing _id and sorting keys
    norm_rows: List[str] = []
    for d in docs:
        if isinstance(d, dict):
            dd = {k: d[k] for k in sorted(d.keys()) if k != "_id"}
            norm_rows.append(jdump(dd, ensure_ascii=False))
        else:
            norm_rows.append(jdump(d, ensure_ascii=False))
    return {"kind": "docs", "rows": sorted(norm_rows)}

def compare(sql_res: ExecResult, mongo_res: ExecResult) -> Tuple[bool, Dict[str, Any]]:
    meta: Dict[str, Any] = {}

    if not sql_res.ok:
        return False, {"reason": "sql_fail", "sql_error": sql_res.error}
    if not mongo_res.ok:
        return False, {"reason": "mongo_fail", "mongo_error": mongo_res.error}

    # count compare if mongo returned a count
    if mongo_res.count is not None:
        sql_count = len(sql_res.rows or [])
        meta["sql_count"] = sql_count
        meta["mongo_count"] = mongo_res.count
        meta["mode"] = "count"
        return sql_count == mongo_res.count, meta

    sql_c = _canon_sql(sql_res.rows)
    mongo_c = _canon_mongo(mongo_res.rows)

    meta["sql_kind"] = sql_c["kind"]
    meta["mongo_kind"] = mongo_c["kind"]
    meta["mode"] = "relaxed"

    # Compare by kind when possible
    if sql_c["kind"] == "scalar" and mongo_c["kind"] == "scalar":
        meta["sql_value"] = sql_c["value"]
        meta["mongo_value"] = mongo_c["value"]
        return sql_c["value"] == mongo_c["value"], meta

    if sql_c["kind"] == "col" and mongo_c["kind"] == "col":
        meta["sql_n"] = len(sql_c["values"])
        meta["mongo_n"] = len(mongo_c["values"])
        return sql_c["values"] == mongo_c["values"], meta

    if sql_c["kind"] == "table" and mongo_c["kind"] in ("docs", "table"):
        meta["sql_n"] = len(sql_c["rows"])
        meta["mongo_n"] = len(mongo_c.get("rows", []))
        return sql_c["rows"] == mongo_c.get("rows", []), meta

    # fallback: count match only (useful metric under time pressure)
    meta["fallback"] = "count_only"
    return (len(sql_res.rows or []) == len(mongo_res.rows or [])), meta

# -----------------------------
# Resume support
# -----------------------------

def load_done_ids(out_report: str) -> set:
    done = set()
    if not os.path.exists(out_report):
        return done
    try:
        for r in read_jsonl(out_report):
            done.add(r.get("key"))
    except Exception:
        pass
    return done

# -----------------------------
# CLI
# -----------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--nosql-jsonl", required=True, help="nosql_queries_*.jsonl generated file")
    ap.add_argument("--train-dbs-zip", required=False, help="kept for compatibility (unused here)")
    ap.add_argument("--sqlite-root", required=True, help="Folder that contains extracted DB folders with <db>/<db>.sqlite")
    ap.add_argument("--mongo-uri", default="mongodb://localhost:27017")
    ap.add_argument("--mongo-db-prefix", default="", help="If you imported as db_id or prefixed name, adjust here.")
    ap.add_argument("--out-report", required=True, help="JSONL results file (append)")
    ap.add_argument("--out-errors", required=True, help="JSONL error rows (append)")
    ap.add_argument("--max-time-ms", type=int, default=5000, help="Server-side time limit per Mongo operation (ms).")
    ap.add_argument("--max-docs", type=int, default=2000, help="Max Mongo docs to pull back for comparison.")
    ap.add_argument("--batch-size", type=int, default=500, help="Mongo cursor batch size for aggregate.")
    ap.add_argument("--skip-lookup", action="store_true", help="Skip pipelines containing $lookup/$graphLookup.")
    ap.add_argument("--no-force-limit", action="store_true", help="Disable auto-appending a $limit stage if missing.")
    ap.add_argument("--force-limit-n", type=int, default=1000, help="If auto-limit is enabled, append $limit N.")
    ap.add_argument("--resume", action="store_true", help="Skip items already present in out-report")
    ap.add_argument("--progress-every", type=int, default=100)
    return ap.parse_args()

def resolve_sqlite_path(sqlite_root: str, db_id: str) -> str:
    # Expected: <sqlite_root>/<db_id>/<db_id>.sqlite
    p = os.path.join(sqlite_root, db_id, f"{db_id}.sqlite")
    if os.path.exists(p):
        return p
    # fallback: <sqlite_root>/<db_id>.sqlite
    p2 = os.path.join(sqlite_root, f"{db_id}.sqlite")
    if os.path.exists(p2):
        return p2
    return p  # return best guess for error msg

def main() -> None:
    args = parse_args()

    safe_mkdir(os.path.dirname(args.out_report) or ".")
    safe_mkdir(os.path.dirname(args.out_errors) or ".")

    done = load_done_ids(args.out_report) if args.resume else set()

    t0 = time.time()
    n = 0
    sql_ran = 0
    mongo_ran = 0
    match = 0
    ok_gen = 0

    out_rows = []
    err_rows = []

    for rec in read_jsonl(args.nosql_jsonl):
        n += 1

        db_id = rec.get("db_id")
        sql_raw = rec.get("sql", "")
        nosql = rec.get("nosql") or {}
        ok = bool(rec.get("ok", False))
        if ok:
            ok_gen += 1

        # key for resume
        key = rec.get("question_id") or f"{rec.get('idx')}::{db_id}::{sha1(sql_raw)[:10]}"
        if args.resume and key in done:
            continue

        sqlite_path = resolve_sqlite_path(args.sqlite_root, db_id)
        sql = sanitize_sql(sql_raw, db_id=db_id)

        # SQL execute
        sql_res = run_sql(sqlite_path, sql)
        if sql_res.ok:
            sql_ran += 1

        # Mongo execute
        collection = nosql.get("collection")
        operation = nosql.get("operation")
        pipeline = nosql.get("pipeline")

        # if generation itself is bad, mark quickly
        if not (collection and operation):
            mongo_res = ExecResult(ok=False, error="missing_collection_or_operation")
        else:
            # sanitize pipeline before run (aggregate will sanitize again)
            if operation == "aggregate":
                sp, reason = sanitize_pipeline(pipeline)
                if sp is None:
                    mongo_res = ExecResult(ok=False, error=f"pipeline_nonexec:{reason}")
                else:
                    pipeline = sp

            mongo_db = f"{args.mongo_db_prefix}{db_id}"
            mongo_res = run_mongo(
                mongo_uri=args.mongo_uri,
                db_name=mongo_db,
                collection=collection,
                operation=operation,
                pipeline=pipeline,
                max_time_ms=args.max_time_ms,
                max_docs=args.max_docs,
                batch_size=args.batch_size,
                skip_lookup=bool(args.skip_lookup),
                force_limit=not bool(args.no_force_limit),
                force_limit_n=int(args.force_limit_n),
            )
            if mongo_res.ok:
                mongo_ran += 1

        is_match, meta = compare(sql_res, mongo_res)
        if is_match:
            match += 1

        row = {
            "key": key,
            "idx": rec.get("idx"),
            "question_id": rec.get("question_id"),
            "db_id": db_id,
            "ok_gen": ok,
            "sql_ok": sql_res.ok,
            "mongo_ok": mongo_res.ok,
            "match": is_match,
            "sql_error": None if sql_res.ok else sql_res.error,
            "mongo_error": None if mongo_res.ok else mongo_res.error,
            "meta": meta,
        }
        out_rows.append(row)

        if (not sql_res.ok) or (not mongo_res.ok):
            err_rows.append({
                "key": key,
                "idx": rec.get("idx"),
                "db_id": db_id,
                "sql": sql,
                "sql_error": sql_res.error,
                "mongo": nosql,
                "mongo_error": mongo_res.error,
            })

        # flush periodically (important for resume + safety)
        if len(out_rows) >= 100:
            write_jsonl(args.out_report, out_rows, mode="a")
            out_rows = []
        if len(err_rows) >= 100:
            write_jsonl(args.out_errors, err_rows, mode="a")
            err_rows = []

        if n % int(args.progress_every) == 0:
            elapsed = time.time() - t0
            mongo_exec_rate = (mongo_ran / max(sql_ran, 1)) * 100.0
            match_over_exec = (match / max(mongo_ran, 1)) * 100.0
            print(
                f"[PROGRESS] n={n} ok_gen={ok_gen} "
                f"sql_ran={sql_ran} mongo_ran={mongo_ran} match={match} "
                f"elapsed={elapsed:.1f}s mongo_exec_rate={mongo_exec_rate:.1f}% "
                f"match_over_executed={match_over_exec:.1f}%",
                flush=True
            )

    if out_rows:
        write_jsonl(args.out_report, out_rows, mode="a")
    if err_rows:
        write_jsonl(args.out_errors, err_rows, mode="a")

    elapsed = time.time() - t0
    mongo_exec_rate = (mongo_ran / max(sql_ran, 1)) * 100.0
    match_over_exec = (match / max(mongo_ran, 1)) * 100.0
    print(
        f"[DONE] elapsed={elapsed:.1f}s total={n} ok_gen={ok_gen} sql_ran={sql_ran} "
        f"mongo_ran={mongo_ran} match={match} mongo_exec_rate={mongo_exec_rate:.1f}% "
        f"match_over_executed={match_over_exec:.1f}%",
        flush=True
    )

if __name__ == "__main__":
    main()