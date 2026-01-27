#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
import os
import random
import re
import time
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr

# --- Optional: execution evaluation helpers (SQL vs Mongo) ---
try:
    from verify_exec_accuracy_optimized import run_sql as _run_sql_eval, run_mongo as _run_mongo_eval
except Exception:
    _run_sql_eval = None
    _run_mongo_eval = None

import requests

try:
    import rag_retrieve_fast_v2 as rag_retrieve  # fastest path
except Exception:
    try:
        import rag_retrieve_fast as rag_retrieve  # fallback fast
    except Exception:
        import rag_retrieve_fixed as rag_retrieve  # fallback stable


DEFAULT_OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
DEFAULT_SQL_MODEL = os.getenv("SQL_MODEL", "qwen2.5:latest")
DEFAULT_MONGO_MODEL = os.getenv("MONGO_MODEL", "qwen2.5:latest")


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def extract_json_candidate(text: str) -> str:
    """
    Extract the most likely JSON object/array from a text blob that may include code fences
    or surrounding commentary.
    """
    if text is None:
        return ""
    s = text.strip()

    # Remove code fences if present
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```$", "", s)

    # Try to find first { ... } or [ ... ]
    # Use a simple greedy approach: take substring from first '{' to last '}' if exists;
    # else from first '[' to last ']'.
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
    """
    Minimal cleanup:
    - strip trailing commas before } or ]
    - remove non-printable characters
    """
    if s is None:
        return ""
    s = "".join(ch for ch in s if ch.isprintable() or ch in "\n\t\r")
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s.strip()


def call_ollama_generate(ollama_url: str, model: str, prompt: str, temperature: float = 0.0) -> str:
    """
    Calls Ollama /api/generate (streaming false).
    """
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
    # Canonical output format your pipeline expects
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

{("Few-shot Examples:\\n" + fewshot) if fewshot else ""}

SQL:
{sql}
"""


@lru_cache(maxsize=512)
def load_text_file(path: str) -> str:
    if not path:
        return ""
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_records_jsonl(path: str, max_n: int = 5000) -> List[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return []
    out = []
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


def run_pipeline(
    question: str,
    db_id: str,
    ollama_url: str,
    sql_model: str,
    mongo_model: str,
    fewshot_path: str,
    fewshot_k: int,
    use_fewshot_mongo: bool,
) -> Tuple[str, str, str]:
    """
    Main generation pipeline:
      1) retrieval context
      2) text->sql
      3) sql->mongo json
    Returns: (sql, mongo_json, debug_text)
    """
    t0 = time.time()
    debug = []
    debug.append(f"[{now_ts()}] Starting pipeline")
    debug.append(f"DB: {db_id}")

    # Retrieve context
    try:
        ctx = rag_retrieve.get_context(question=question, db_id=db_id, top_k=max(8, fewshot_k))
        schema_context = ctx.get("context", "")
        hits = ctx.get("hits", [])
        debug.append(f"RAG hits: {len(hits)}")
    except Exception as e:
        schema_context = ""
        debug.append(f"RAG retrieval failed: {e}")

    # SQL generation
    sql_prompt = make_sql_prompt(question, schema_context)
    try:
        sql = call_ollama_generate(ollama_url, sql_model, sql_prompt, temperature=0.0).strip()
    except Exception as e:
        sql = ""
        debug.append(f"SQL generation failed: {e}")

    # Few-shot (optional)
    fewshot = ""
    if use_fewshot_mongo and fewshot_path:
        fewshot = load_text_file(fewshot_path)

    # Mongo generation
    mongo_prompt = make_mongo_prompt(sql, schema_context, fewshot=fewshot)
    try:
        mongo_raw = call_ollama_generate(ollama_url, mongo_model, mongo_prompt, temperature=0.0)
        mongo_candidate = extract_json_candidate(mongo_raw)
        mongo_candidate = clean_json_string(mongo_candidate)
        # Validate JSON
        _ = json.loads(mongo_candidate)
        mongo_json = mongo_candidate
    except Exception as e:
        mongo_json = ""
        debug.append(f"Mongo generation/parse failed: {e}")

    dt = time.time() - t0
    debug.append(f"Done in {dt:.2f}s")
    return sql, mongo_json, "\n".join(debug)


def regenerate_mongo_only(
    db_id: str,
    sql_text: str,
    ollama_url: str,
    mongo_model: str,
    fewshot_path: str,
    fewshot_k: int,
    use_fewshot_mongo: bool,
) -> Tuple[str, str]:
    """
    Regenerate only Mongo output for a given SQL and DB context.
    """
    debug = []
    debug.append(f"[{now_ts()}] Regenerate Mongo only")
    try:
        ctx = rag_retrieve.get_context(question=sql_text, db_id=db_id, top_k=max(8, fewshot_k))
        schema_context = ctx.get("context", "")
        hits = ctx.get("hits", [])
        debug.append(f"RAG hits: {len(hits)}")
    except Exception as e:
        schema_context = ""
        debug.append(f"RAG retrieval failed: {e}")

    fewshot = ""
    if use_fewshot_mongo and fewshot_path:
        fewshot = load_text_file(fewshot_path)

    mongo_prompt = make_mongo_prompt(sql_text, schema_context, fewshot=fewshot)

    try:
        mongo_raw = call_ollama_generate(ollama_url, mongo_model, mongo_prompt, temperature=0.0)
        mongo_candidate = extract_json_candidate(mongo_raw)
        mongo_candidate = clean_json_string(mongo_candidate)
        _ = json.loads(mongo_candidate)
        return mongo_candidate, "\n".join(debug)
    except Exception as e:
        debug.append(f"Mongo regen failed: {e}")
        return "", "\n".join(debug)


def repair_mongo_json(mongo_text: str) -> Tuple[str, str]:
    """
    Best-effort "repair": extract JSON candidate + clean trailing commas.
    """
    debug = []
    debug.append(f"[{now_ts()}] Repair Mongo JSON")
    cand = extract_json_candidate(mongo_text or "")
    cand = clean_json_string(cand)
    try:
        _ = json.loads(cand)
        debug.append("JSON valid after cleanup.")
        return cand, "\n".join(debug)
    except Exception as e:
        debug.append(f"Still invalid JSON: {e}")
        return cand, "\n".join(debug)


def run_execution_compare(db_id: str,
                          sql_text: str,
                          mongo_text: str,
                          sqlite_root: str,
                          mongo_uri: str,
                          limit_preview: int = 10):
    """Run SQL on SQLite DB and Mongo pipeline on MongoDB, then compare outputs.

    Returns (markdown_summary, sql_preview_json, mongo_preview_json).
    """
    # Basic guards
    db_id = (db_id or "").strip()
    if not db_id:
        return "❌ **DB ID is required** (e.g., `financial`, `superstore`).", None, None

    if _run_sql_eval is None or _run_mongo_eval is None:
        return ("❌ **Execution helpers not available.**\n\n"
                "Make sure `verify_exec_accuracy_optimized.py` is in the same folder as this app "
                "and dependencies (`pymongo`) are installed."), None, None

    sqlite_root = (sqlite_root or "").strip()
    mongo_uri = (mongo_uri or "").strip()
    if not sqlite_root:
        return "❌ **SQLite root path is empty.** Point it to your `train_databases/` folder.", None, None
    if not mongo_uri:
        return "❌ **Mongo URI is empty.** Example: `mongodb://localhost:27017`", None, None

    # Parse Mongo JSON
    try:
        mongo_candidate = extract_json_candidate(mongo_text or "")
        mongo_candidate = clean_json_string(mongo_candidate)
        mongo_obj = json.loads(mongo_candidate)
    except Exception as e:
        return f"❌ **Mongo JSON parse failed**: {e}", None, None

    # Run SQL
    try:
        sql_rows = _run_sql_eval(sqlite_root=sqlite_root, db_id=db_id, sql=sql_text)
        sql_err = None
    except Exception as e:
        sql_rows, sql_err = None, str(e)

    # Run Mongo
    try:
        mongo_rows = _run_mongo_eval(mongo_uri=mongo_uri, db_id=db_id, mongo_obj=mongo_obj)
        mongo_err = None
    except Exception as e:
        mongo_rows, mongo_err = None, str(e)

    # Summaries
    def _preview(rows):
        if rows is None:
            return None
        if isinstance(rows, dict):
            return rows
        # list of rows/tuples/dicts
        out = []
        for r in rows[:limit_preview]:
            if isinstance(r, (list, dict, str, int, float)) or r is None:
                out.append(r)
            else:
                # tuple / sqlite row
                try:
                    out.append(list(r))
                except Exception:
                    out.append(str(r))
        return out

    sql_prev = _preview(sql_rows) if sql_err is None else None
    mongo_prev = _preview(mongo_rows) if mongo_err is None else None

    # Compare (simple): count + normalized string set for first N
    match_note = "N/A"
    if sql_err is None and mongo_err is None:
        try:
            sql_norm = [json.dumps(x, sort_keys=True, default=str) for x in (sql_prev or [])]
            mongo_norm = [json.dumps(x, sort_keys=True, default=str) for x in (mongo_prev or [])]
            same_count = (len(sql_rows) == len(mongo_rows)) if isinstance(sql_rows, list) and isinstance(mongo_rows, list) else False
            overlap = len(set(sql_norm).intersection(set(mongo_norm)))
            match_note = f"Count match: **{same_count}**; Preview overlap (first {limit_preview}): **{overlap}**"
        except Exception:
            match_note = "Computed basic preview comparison."

    md = []
    md.append(f"### Execution Results (DB: `{db_id}`)")
    if sql_err:
        md.append(f"- **SQL**: ❌ Failed — `{sql_err}`")
    else:
        md.append(f"- **SQL**: ✅ OK — rows: **{len(sql_rows) if isinstance(sql_rows, list) else 'n/a'}**")
    if mongo_err:
        md.append(f"- **Mongo**: ❌ Failed — `{mongo_err}`")
    else:
        md.append(f"- **Mongo**: ✅ OK — rows: **{len(mongo_rows) if isinstance(mongo_rows, list) else 'n/a'}**")
    md.append(f"- **Quick Compare**: {match_note}")
    md.append("\n> Tip: exact string/pipeline match is strict; execution-based comparison is the meaningful signal.")
    return "\n".join(md), sql_prev, mongo_prev


def build_ui():
    with gr.Blocks(title="Text→SQL→Mongo (Fast + Accurate)", theme=gr.themes.Default()) as demo:
        gr.Markdown("# Text→SQL→Mongo (Hybrid RAG + Qwen2.5)\n\nGenerate **SQL** and **MongoDB pipeline JSON**, then optionally run **execution**.")

        with gr.Row():
            question = gr.Textbox(label="User question", placeholder="Ask a question…", lines=2)
            db_id = gr.Textbox(label="DB ID", placeholder="financial / superstore / ...", value="financial")

        with gr.Row():
            ollama_url = gr.Textbox(label="Ollama URL", value=DEFAULT_OLLAMA_URL)
            sql_model = gr.Textbox(label="Text-to-SQL model", value=DEFAULT_SQL_MODEL)
            mongo_model = gr.Textbox(label="SQL-to-Mongo model", value=DEFAULT_MONGO_MODEL)

        with gr.Accordion("Few-shot options", open=False):
            fewshot_path = gr.Textbox(label="Few-shot file path (optional)", value="")
            fewshot_k = gr.Slider(label="Few-shot top-k (for retrieval)", minimum=0, maximum=20, value=8, step=1)
            use_fewshot_mongo = gr.Checkbox(label="Use few-shot in SQL→Mongo prompt", value=True)

        with gr.Row():
            run_btn = gr.Button("Run (Text→SQL→Mongo)", variant="primary")
            regen_mongo_btn = gr.Button("Regenerate Mongo only", variant="secondary")
            repair_btn = gr.Button("Repair Mongo JSON", variant="secondary")

        gr.Markdown("## Outputs")
        sql_out = gr.Textbox(label="Generated SQL", lines=4)
        mongo_out = gr.Textbox(label="Generated Mongo JSON", lines=16)
        debug_out = gr.Textbox(label="Debug (RAG hits, paths)", lines=20)

        # --- Execution (SQL vs Mongo) ---
        with gr.Accordion("Execution (run SQL + run Mongo and compare)", open=False):
            sqlite_root = gr.Textbox(
                label="SQLite DB root (folder containing <db_id>/<db_id>.sqlite)",
                value=os.getenv("BIRD_SQLITE_ROOT", "./train_databases"),
            )
            mongo_uri = gr.Textbox(
                label="MongoDB URI",
                value=os.getenv("MONGO_URI", "mongodb://localhost:27017"),
            )
            exec_btn = gr.Button("Run execution (SQL vs Mongo)", variant="primary")
            exec_md = gr.Markdown()
            with gr.Row():
                sql_preview = gr.JSON(label="SQL Result Preview")
                mongo_preview = gr.JSON(label="Mongo Result Preview")

        run_btn.click(
            fn=run_pipeline,
            inputs=[question, db_id, ollama_url, sql_model, mongo_model, fewshot_path, fewshot_k, use_fewshot_mongo],
            outputs=[sql_out, mongo_out, debug_out],
        )

        regen_mongo_btn.click(
            fn=regenerate_mongo_only,
            inputs=[db_id, sql_out, ollama_url, mongo_model, fewshot_path, fewshot_k, use_fewshot_mongo],
            outputs=[mongo_out, debug_out],
        )

        # Bind execution compare
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
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7863)
    args = parser.parse_args()

    demo = build_ui()
    demo.launch(server_name=args.host, server_port=args.port)


if __name__ == "__main__":
    main()

