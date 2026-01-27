#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio App (v3.1) — Text → SQL → Mongo + Execution + Metrics (port 7863)

FIX for your crash:
- gr.JSON expects a Python dict/list (or a JSON string). When it receives None/""/plain text,
  Gradio tries to parse it as JSON and fails with:
  orjson.JSONDecodeError: unexpected character: line 1 column 1

So this version GUARANTEES that JSON outputs are ALWAYS dict/list:
- On error: returns {} (empty dict) instead of None
- Previews are coerced to dict/list safely

Also keeps the earlier fixes:
- sqlite_path resolved as: <sqlite_root>/<db_id>/<db_id>.sqlite
- Calls verify_exec_accuracy_optimized.run_sql/run_mongo with only supported kwargs
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import re
import time
from functools import lru_cache
from typing import Any, Dict, List, Tuple

import gradio as gr
import requests

# --------------------------
# Defaults for YOUR machine
# --------------------------
DEFAULT_TRAIN_ROOT = os.getenv("TRAIN_ROOT", "/Users/pavanpratyusha/Desktop/train")
DEFAULT_BIRD_OUTPUTS = os.getenv("BIRD_OUTPUTS", "/Users/pavanpratyusha/Desktop/bird_outputs")

DEFAULT_RAG_INDEX = os.getenv(
    "RAG_INDEX_PATH",
    os.path.join(DEFAULT_BIRD_OUTPUTS, "rag_index.sqlite"),
)

DEFAULT_FEWSHOT_JSONL = os.getenv(
    "FEWSHOT_JSONL",
    os.path.join(DEFAULT_BIRD_OUTPUTS, "fewshot_ok", "fewshot_bank.jsonl"),
)

DEFAULT_SQLITE_ROOT = os.getenv(
    "BIRD_SQLITE_ROOT",
    os.path.join(DEFAULT_TRAIN_ROOT, "train_databases"),
)

DEFAULT_MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_SQL_MODEL = os.getenv("SQL_MODEL", "qwen2.5-coder:3b-instruct")
DEFAULT_MONGO_MODEL = os.getenv("MONGO_MODEL", "qwen2.5-coder:3b-instruct")


# --------------------------
# Small helpers
# --------------------------
def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _path_exists(p: str) -> str:
    return "✅" if p and os.path.exists(p) else "❌"


def extract_json_candidate(text: str) -> str:
    """Extract likely JSON object/array from text (strip code fences, take {..} or [..])."""
    if text is None:
        return ""
    s = text.strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    if "{" in s and "}" in s:
        start = s.find("{")
        end = s.rfind("}")
        if end > start:
            return s[start : end + 1]

    if "[" in s and "]" in s:
        start = s.find("[")
        end = s.rfind("]")
        if end > start:
            return s[start : end + 1]

    return s


def clean_json_string(s: str) -> str:
    """Minimal cleanup: remove non-printables + trailing commas before } or ]."""
    if s is None:
        return ""
    s = "".join(ch for ch in s if ch.isprintable() or ch in "\n\t\r")
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s.strip()


def call_ollama_generate(ollama_url: str, model: str, prompt: str, temperature: float = 0.0) -> str:
    """Calls Ollama /api/generate (non-stream)."""
    url = ollama_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")


def make_sql_prompt(question: str, schema_context: str) -> str:
    return f"""You are a Text-to-SQL system.
Generate a single SQLite-compatible SQL query.

Question:
{question}

Schema / Context:
{schema_context}

Return ONLY SQL (no markdown, no explanation).
"""


def make_mongo_prompt(sql: str, schema_context: str, fewshot: str = "") -> str:
    return f"""You are a SQL-to-MongoDB translator.

Convert the following SQL into a MongoDB query JSON in this EXACT structure:

{{
  "collection": "<collection_name>",
  "operation": "aggregate",
  "pipeline": [ ... ]
}}

Rules:
- Output ONLY valid JSON. No markdown. No comments.
- Use MongoDB aggregation pipeline when needed.
- Prefer $lookup for joins, $group for aggregation, $project for select/rename.
- Keep field names as per schema context.

Schema / Context:
{schema_context}

{("Few-shot Examples (JSONL):\\n" + fewshot) if fewshot else ""}

SQL:
{sql}
"""


@lru_cache(maxsize=256)
def load_records_jsonl(path: str, max_n: int = 200000) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_n:
                break
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def format_fewshot_jsonl(records: List[Dict[str, Any]], k: int, db_id: str) -> str:
    """Pick k ok examples; prefer matching db_id if present in jsonl."""
    if not records or k <= 0:
        return ""
    db_id = (db_id or "").strip()

    same_db = [r for r in records if str(r.get("db_id", "")).strip() == db_id]
    pool = same_db if len(same_db) >= max(3, k // 2) else records

    ok_pool = [r for r in pool if r.get("ok") is True] or pool

    sample = random.sample(ok_pool, k=min(k, len(ok_pool)))
    formatted_lines = []
    for r in sample:
        obj = {
            "question": r.get("question") or r.get("nl") or r.get("text") or "",
            "sql": r.get("sql") or "",
            "mongo": r.get("mongo") or r.get("nosql") or r.get("pipeline") or r.get("mongo_json") or "",
        }
        formatted_lines.append(json.dumps(obj, ensure_ascii=False))
    return "\n".join(formatted_lines)


# --------------------------
# Robust signature adapter
# --------------------------
def call_with_supported_kwargs(fn, **kwargs):
    """
    Call fn(**kwargs) but drop kwargs that fn doesn't accept.
    Returns: (result, used_kwargs)
    """
    sig = inspect.signature(fn)
    accepted = set(sig.parameters.keys())
    filtered = {k: v for k, v in kwargs.items() if k in accepted}
    return fn(**filtered), filtered


# --------------------------
# JSON-safe coercion for gr.JSON
# --------------------------
def to_jsonable(value: Any) -> Any:
    """
    Ensure return value is dict/list for gr.JSON.
    - None -> {}
    - str -> {"_text": "..."}  (NOT parsed as JSON)
    - scalar -> {"_value": "..."}
    - tuple/set -> list(...)
    """
    if value is None:
        return {}
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, (tuple, set)):
        return list(value)
    if isinstance(value, (int, float, bool)):
        return {"_value": value}
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return {}
        # If it's already a valid JSON string, parse it to dict/list
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, (dict, list)):
                    return parsed
            except Exception:
                pass
        return {"_text": value}
    # fallback: stringify
    return {"_value": str(value)}


def preview_rows(rows: Any, limit_preview: int = 10) -> Any:
    """
    Produce a JSONable preview of query result.
    - list -> first N items (each item coerced)
    - dict -> dict coerced
    - other -> coerced scalar
    """
    if rows is None:
        return {}
    if isinstance(rows, dict):
        # try to reduce huge dicts
        return to_jsonable(rows)
    if isinstance(rows, list):
        out = []
        for r in rows[:limit_preview]:
            out.append(to_jsonable(r))
        return out
    return to_jsonable(rows)


# --------------------------
# RAG import (your modules)
# --------------------------
os.environ.setdefault("TRAIN_ROOT", DEFAULT_TRAIN_ROOT)
os.environ.setdefault("BIRD_OUTPUTS", DEFAULT_BIRD_OUTPUTS)
os.environ.setdefault("RAG_INDEX_PATH", DEFAULT_RAG_INDEX)
os.environ.setdefault("RAG_INDEX", DEFAULT_RAG_INDEX)
os.environ.setdefault("RAG_DB_PATH", DEFAULT_RAG_INDEX)

try:
    import rag_retrieve_fast_v2 as rag_retrieve  # fastest path
except Exception:
    try:
        import rag_retrieve_fast as rag_retrieve  # fallback
    except Exception:
        import rag_retrieve_fixed as rag_retrieve  # stable fallback


# --------------------------
# Execution helpers
# --------------------------
try:
    from verify_exec_accuracy_optimized import run_sql as _run_sql_eval, run_mongo as _run_mongo_eval
except Exception:
    _run_sql_eval = None
    _run_mongo_eval = None


# --------------------------
# Main pipeline
# --------------------------
def run_pipeline(
    question: str,
    db_id: str,
    train_root: str,
    rag_index_path: str,
    ollama_url: str,
    sql_model: str,
    mongo_model: str,
    fewshot_jsonl: str,
    fewshot_k: int,
    use_fewshot_mongo: bool,
) -> Tuple[str, str, str]:
    t0 = time.time()
    debug = []
    debug.append(f"[{now_ts()}] Starting pipeline")
    db_id = (db_id or "").strip()
    debug.append(f"DB: {db_id or '(empty)'}")

    os.environ["TRAIN_ROOT"] = (train_root or DEFAULT_TRAIN_ROOT).strip()
    os.environ["RAG_INDEX_PATH"] = (rag_index_path or DEFAULT_RAG_INDEX).strip()
    os.environ["RAG_INDEX"] = os.environ["RAG_INDEX_PATH"]
    os.environ["RAG_DB_PATH"] = os.environ["RAG_INDEX_PATH"]

    debug.append(f"TRAIN_ROOT: {os.environ['TRAIN_ROOT']} ({_path_exists(os.environ['TRAIN_ROOT'])})")
    debug.append(f"RAG_INDEX_PATH: {os.environ['RAG_INDEX_PATH']} ({_path_exists(os.environ['RAG_INDEX_PATH'])})")
    debug.append(f"FEWSHOT_JSONL: {fewshot_jsonl} ({_path_exists((fewshot_jsonl or '').strip())})")

    schema_context = ""
    try:
        ctx = rag_retrieve.get_context(question=question, db_id=db_id, top_k=max(8, int(fewshot_k or 0)))
        schema_context = ctx.get("context", "") or ""
        hits = ctx.get("hits", []) or []
        debug.append(f"RAG hits: {len(hits)}")
    except Exception as e:
        debug.append(f"RAG retrieval failed: {e}")

    sql_prompt = make_sql_prompt(question, schema_context)
    try:
        sql = call_ollama_generate(ollama_url, sql_model, sql_prompt, temperature=0.0).strip()
    except Exception as e:
        sql = ""
        debug.append(f"SQL generation failed: {e}")

    fewshot = ""
    if use_fewshot_mongo and fewshot_jsonl and int(fewshot_k or 0) > 0:
        recs = load_records_jsonl(fewshot_jsonl.strip())
        fewshot = format_fewshot_jsonl(recs, k=int(fewshot_k), db_id=db_id)
        debug.append(f"Few-shot loaded: {len(recs)} records; using k={min(int(fewshot_k), len(recs))}")

    mongo_prompt = make_mongo_prompt(sql, schema_context, fewshot=fewshot)
    try:
        mongo_raw = call_ollama_generate(ollama_url, mongo_model, mongo_prompt, temperature=0.0)
        mongo_candidate = clean_json_string(extract_json_candidate(mongo_raw))
        _ = json.loads(mongo_candidate)
        mongo_json = mongo_candidate
    except Exception as e:
        mongo_json = ""
        debug.append(f"Mongo generation/parse failed: {e}")

    debug.append(f"Done in {time.time() - t0:.2f}s")
    return sql, mongo_json, "\n".join(debug)


def regenerate_mongo_only(
    db_id: str,
    sql_text: str,
    train_root: str,
    rag_index_path: str,
    ollama_url: str,
    mongo_model: str,
    fewshot_jsonl: str,
    fewshot_k: int,
    use_fewshot_mongo: bool,
) -> Tuple[str, str]:
    debug = [f"[{now_ts()}] Regenerate Mongo only"]

    os.environ["TRAIN_ROOT"] = (train_root or DEFAULT_TRAIN_ROOT).strip()
    os.environ["RAG_INDEX_PATH"] = (rag_index_path or DEFAULT_RAG_INDEX).strip()
    os.environ["RAG_INDEX"] = os.environ["RAG_INDEX_PATH"]
    os.environ["RAG_DB_PATH"] = os.environ["RAG_INDEX_PATH"]

    db_id = (db_id or "").strip()

    try:
        ctx = rag_retrieve.get_context(question=sql_text, db_id=db_id, top_k=max(8, int(fewshot_k or 0)))
        schema_context = ctx.get("context", "") or ""
        hits = ctx.get("hits", []) or []
        debug.append(f"RAG hits: {len(hits)}")
    except Exception as e:
        schema_context = ""
        debug.append(f"RAG retrieval failed: {e}")

    fewshot = ""
    if use_fewshot_mongo and fewshot_jsonl and int(fewshot_k or 0) > 0:
        recs = load_records_jsonl(fewshot_jsonl.strip())
        fewshot = format_fewshot_jsonl(recs, k=int(fewshot_k), db_id=db_id)

    mongo_prompt = make_mongo_prompt(sql_text, schema_context, fewshot=fewshot)
    try:
        mongo_raw = call_ollama_generate(ollama_url, mongo_model, mongo_prompt, temperature=0.0)
        mongo_candidate = clean_json_string(extract_json_candidate(mongo_raw))
        _ = json.loads(mongo_candidate)
        return mongo_candidate, "\n".join(debug)
    except Exception as e:
        debug.append(f"Mongo regen failed: {e}")
        return "", "\n".join(debug)


def repair_mongo_json(mongo_text: str) -> Tuple[str, str]:
    debug = [f"[{now_ts()}] Repair Mongo JSON"]
    cand = clean_json_string(extract_json_candidate(mongo_text or ""))
    try:
        _ = json.loads(cand)
        debug.append("JSON valid after cleanup.")
    except Exception as e:
        debug.append(f"Still invalid JSON: {e}")
    return cand, "\n".join(debug)


def run_execution_compare(
    db_id: str,
    sql_text: str,
    mongo_text: str,
    sqlite_root: str,
    mongo_uri: str,
    limit_preview: int = 10,
):
    """
    Run SQL on SQLite and Mongo pipeline on MongoDB, then compare outputs.
    JSON outputs are ALWAYS dict/list to keep gr.JSON happy.
    """
    db_id = (db_id or "").strip()
    if not db_id:
        return "❌ **DB ID is required** (e.g., `financial`, `superstore`).", {}, {}

    if _run_sql_eval is None or _run_mongo_eval is None:
        return (
            "❌ **Execution helpers not available.**\n\n"
            "Ensure `verify_exec_accuracy_optimized.py` is in the same folder as this app.\n"
            "Also install pymongo: `python3 -m pip install pymongo`",
            {},
            {},
        )

    sqlite_root = (sqlite_root or "").strip()
    mongo_uri = (mongo_uri or "").strip()
    if not sqlite_root:
        return "❌ **SQLite root path is empty.** Point it to `.../train/train_databases`.", {}, {}
    if not mongo_uri:
        return "❌ **Mongo URI is empty.** Example: `mongodb://localhost:27017`", {}, {}

    sqlite_path = os.path.join(sqlite_root, db_id, f"{db_id}.sqlite")

    # Parse Mongo JSON
    try:
        mongo_candidate = clean_json_string(extract_json_candidate(mongo_text or ""))
        mongo_obj = json.loads(mongo_candidate)
    except Exception as e:
        md = [
            f"### Execution Results (DB: `{db_id}`)",
            f"- sqlite_path resolved: `{sqlite_path}` ({_path_exists(sqlite_path)})",
            f"- **Mongo JSON parse failed**: `{e}`",
        ]
        return "\n".join(md), {}, {}

    # Run SQL
    used_sql_kwargs = {}
    sql_rows = None
    sql_err = None
    try:
        sql_rows, used_sql_kwargs = call_with_supported_kwargs(
            _run_sql_eval,
            sqlite_root=sqlite_root,  # some variants
            sqlite_path=sqlite_path,  # your version likely uses this
            db_id=db_id,
            sql=sql_text,
        )
    except Exception as e:
        sql_err = str(e)

    # Run Mongo
    used_mongo_kwargs = {}
    mongo_rows = None
    mongo_err = None
    try:
        mongo_rows, used_mongo_kwargs = call_with_supported_kwargs(
            _run_mongo_eval,
            mongo_uri=mongo_uri,
            db_id=db_id,          # dropped if unsupported
            db_name=db_id,        # some variants use db_name
            mongo_obj=mongo_obj,
            mongo_json=mongo_obj, # some variants use mongo_json
        )
    except Exception as e:
        mongo_err = str(e)

    sql_prev = preview_rows(sql_rows, limit_preview=limit_preview) if not sql_err else {}
    mongo_prev = preview_rows(mongo_rows, limit_preview=limit_preview) if not mongo_err else {}

    match_note = "N/A"
    if not sql_err and not mongo_err and isinstance(sql_rows, list) and isinstance(mongo_rows, list):
        try:
            sql_norm = [json.dumps(x, sort_keys=True, default=str) for x in (sql_rows[:limit_preview])]
            mongo_norm = [json.dumps(x, sort_keys=True, default=str) for x in (mongo_rows[:limit_preview])]
            same_count = (len(sql_rows) == len(mongo_rows))
            overlap = len(set(sql_norm).intersection(set(mongo_norm)))
            match_note = f"Count match: **{same_count}**; Preview overlap (first {limit_preview}): **{overlap}**"
        except Exception:
            match_note = "Preview comparison computed (best-effort)."

    md = []
    md.append(f"### Execution Results (DB: `{db_id}`)")
    md.append(f"- SQLite root: `{sqlite_root}` ({_path_exists(sqlite_root)})")
    md.append(f"- sqlite_path resolved: `{sqlite_path}` ({_path_exists(sqlite_path)})")
    md.append(f"- Mongo URI: `{mongo_uri}`")
    md.append(f"- SQL args used: `{used_sql_kwargs}`")
    md.append(f"- Mongo args used: `{used_mongo_kwargs}`")

    if sql_err:
        md.append(f"- **SQL**: ❌ Failed — `{sql_err}`")
    else:
        md.append(f"- **SQL**: ✅ OK — rows: **{len(sql_rows) if isinstance(sql_rows, list) else 'n/a'}**")

    if mongo_err:
        md.append(f"- **Mongo**: ❌ Failed — `{mongo_err}`")
    else:
        md.append(f"- **Mongo**: ✅ OK — rows: **{len(mongo_rows) if isinstance(mongo_rows, list) else 'n/a'}**")

    md.append(f"- **Quick Compare**: {match_note}")
    md.append("\n> Tip: strict JSON/pipeline match is too harsh; execution-based match is the real metric.")

    return "\n".join(md), to_jsonable(sql_prev), to_jsonable(mongo_prev)


# --------------------------
# UI
# --------------------------
def build_ui():
    with gr.Blocks(title="Text→SQL→Mongo (Exec + Metrics)", theme=gr.themes.Default()) as demo:
        gr.Markdown(
            "# Text→SQL→Mongo (Hybrid RAG + Qwen2.5)\n"
            "Generate **SQL** and **MongoDB pipeline JSON**, then optionally run **execution**.\n\n"
            f"**Defaults:**\n"
            f"- TRAIN_ROOT: `{DEFAULT_TRAIN_ROOT}`\n"
            f"- RAG_INDEX: `{DEFAULT_RAG_INDEX}`\n"
            f"- FEWSHOT: `{DEFAULT_FEWSHOT_JSONL}`\n"
            f"- SQLITE_ROOT: `{DEFAULT_SQLITE_ROOT}`\n"
        )

        with gr.Row():
            question = gr.Textbox(label="User question", placeholder="Ask a question…", lines=2)
            db_id = gr.Textbox(label="DB ID", placeholder="financial / superstore / ...", value="superstore")

        with gr.Row():
            train_root = gr.Textbox(label="TRAIN_ROOT", value=DEFAULT_TRAIN_ROOT)
            rag_index_path = gr.Textbox(label="RAG_INDEX_PATH", value=DEFAULT_RAG_INDEX)

        with gr.Row():
            ollama_url = gr.Textbox(label="Ollama URL", value=DEFAULT_OLLAMA_URL)
            sql_model = gr.Textbox(label="Text→SQL model", value=DEFAULT_SQL_MODEL)
            mongo_model = gr.Textbox(label="SQL→Mongo model", value=DEFAULT_MONGO_MODEL)

        with gr.Accordion("Few-shot options", open=False):
            fewshot_jsonl = gr.Textbox(label="Few-shot JSONL path", value=DEFAULT_FEWSHOT_JSONL)
            fewshot_k = gr.Slider(label="Few-shot k", minimum=0, maximum=20, value=3, step=1)
            use_fewshot_mongo = gr.Checkbox(label="Use few-shot in SQL→Mongo prompt", value=True)

        with gr.Row():
            run_btn = gr.Button("Run (Text→SQL→Mongo)", variant="primary")
            regen_mongo_btn = gr.Button("Regenerate Mongo only", variant="secondary")
            repair_btn = gr.Button("Repair Mongo JSON", variant="secondary")

        gr.Markdown("## Outputs")
        sql_out = gr.Textbox(label="Generated SQL", lines=4)
        mongo_out = gr.Textbox(label="Generated Mongo JSON", lines=16)
        debug_out = gr.Textbox(label="Debug", lines=18)

        with gr.Accordion("Execution (run SQL + run Mongo and compare)", open=False):
            sqlite_root = gr.Textbox(
                label="SQLite DB root (folder containing <db_id>/<db_id>.sqlite)",
                value=DEFAULT_SQLITE_ROOT,
            )
            mongo_uri = gr.Textbox(label="MongoDB URI", value=DEFAULT_MONGO_URI)
            exec_btn = gr.Button("Run execution (SQL vs Mongo)", variant="primary")
            exec_md = gr.Markdown()
            with gr.Row():
                sql_preview = gr.JSON(label="SQL Result Preview", value={})
                mongo_preview = gr.JSON(label="Mongo Result Preview", value={})

        run_btn.click(
            fn=run_pipeline,
            inputs=[
                question, db_id,
                train_root, rag_index_path,
                ollama_url, sql_model, mongo_model,
                fewshot_jsonl, fewshot_k, use_fewshot_mongo
            ],
            outputs=[sql_out, mongo_out, debug_out],
        )

        regen_mongo_btn.click(
            fn=regenerate_mongo_only,
            inputs=[
                db_id, sql_out,
                train_root, rag_index_path,
                ollama_url, mongo_model,
                fewshot_jsonl, fewshot_k, use_fewshot_mongo
            ],
            outputs=[mongo_out, debug_out],
        )

        exec_btn.click(
            fn=run_execution_compare,
            inputs=[db_id, sql_out, mongo_out, sqlite_root, mongo_uri],
            outputs=[exec_md, sql_preview, mongo_preview],
        )

        repair_btn.click(
            fn=repair_mongo_json,
            inputs=[mongo_out],
            outputs=[mongo_out, debug_out],
        )

    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7863)

    parser.add_argument("--train-root", default=DEFAULT_TRAIN_ROOT)
    parser.add_argument("--rag-index", default=DEFAULT_RAG_INDEX)
    parser.add_argument("--fewshot", default=DEFAULT_FEWSHOT_JSONL)
    parser.add_argument("--sqlite-root", default=DEFAULT_SQLITE_ROOT)
    args = parser.parse_args()

    os.environ["TRAIN_ROOT"] = args.train_root
    os.environ["RAG_INDEX_PATH"] = args.rag_index
    os.environ["RAG_INDEX"] = args.rag_index
    os.environ["RAG_DB_PATH"] = args.rag_index
    os.environ["FEWSHOT_JSONL"] = args.fewshot
    os.environ["BIRD_SQLITE_ROOT"] = args.sqlite_root

    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()

