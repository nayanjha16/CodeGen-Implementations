#!/usr/bin/env python3
"""
app_gradio_v2_fast_accurate_nowarmup.py

Gradio demo app for:
Text question -> (optional) RAG context -> SQL-to-Mongo pipeline generation
Backends:
  1) Ollama (Qwen2.5 / Qwen2.5-Coder)
  2) MLX fused model (LoRA fused folder from mlx_lm_lora.train)

Supports:
- Optional schema-aware retrieval via rag_retrieve_fast_v2.py (if available)
- Pipeline sanitizer + JSON validity checks
- "Before vs After" style demo: switch backend to show LoRA improvements

Run:
  python app_gradio_v2_fast_accurate_nowarmup.py

Deps:
  pip install gradio requests orjson

Optional (MLX):
  pip install mlx-lm
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import gradio as gr
import requests

try:
    import orjson as _orjson  # faster JSON
except Exception:
    _orjson = None

# -----------------------------
# Optional: your retrieval module
# -----------------------------
RAG_AVAILABLE = False
_rag_mod = None
try:
    import rag_retrieve_fast_v2 as _rag_mod  # expects your file to exist locally
    RAG_AVAILABLE = True
except Exception:
    _rag_mod = None
    RAG_AVAILABLE = False

# -----------------------------
# Optional: MLX generation
# -----------------------------
MLX_AVAILABLE = False
try:
    # mlx-lm provides generate utilities; exact API differs across versions.
    # We'll use a safe subprocess fallback if import fails.
    from mlx_lm import load as mlx_load  # type: ignore
    from mlx_lm import generate as mlx_generate  # type: ignore
    MLX_AVAILABLE = True
except Exception:
    MLX_AVAILABLE = False


# -----------------------------
# Config
# -----------------------------
DEFAULT_OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5")  # or qwen2.5-coder
DEFAULT_DB_ID = os.environ.get("DB_ID", "financial")  # just a default placeholder

# If you have a fused model folder created by mlx_lm_lora.train (it writes a README too)
DEFAULT_MLX_MODEL_DIR = os.environ.get(
    "MLX_MODEL_DIR",
    "adapters_qwen25coder3b_sql2mongo_tiny_1600"
)

# -----------------------------
# Utilities
# -----------------------------
def _dumps(obj: Any) -> str:
    if _orjson:
        return _orjson.dumps(obj, option=_orjson.OPT_INDENT_2).decode("utf-8")
    return json.dumps(obj, indent=2, ensure_ascii=False)

def _loads(s: str) -> Any:
    if _orjson:
        return _orjson.loads(s)
    return json.loads(s)

def now_ms() -> int:
    return int(time.time() * 1000)

def strip_code_fences(text: str) -> str:
    # remove ```json ... ``` or ``` ... ```
    text = text.strip()
    text = re.sub(r"^```(?:json|javascript|js)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()

def extract_first_json_obj(text: str) -> Optional[dict]:
    """
    Tries to extract the first JSON object from messy LLM output.
    Returns dict if found else None.
    """
    t = strip_code_fences(text)

    # Quick path: whole thing is JSON
    try:
        v = _loads(t)
        if isinstance(v, dict):
            return v
    except Exception:
        pass

    # Heuristic: find first { ... } balanced
    start = t.find("{")
    if start == -1:
        return None

    depth = 0
    for i in range(start, len(t)):
        ch = t[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                chunk = t[start:i+1]
                try:
                    v = _loads(chunk)
                    if isinstance(v, dict):
                        return v
                except Exception:
                    return None
    return None

def normalize_pipeline_obj(obj: dict) -> dict:
    """
    Normalizes common field names:
      from/collection
      pipeline/stages
      aggregate/find
    Expected final format (flexible):
      {
        "collection": "...",
        "operation": "aggregate"|"find"|"count"|...,
        "pipeline": [ ... ]  # for aggregate
        "filter": { ... }    # for find
        "projection": { ... }
        "sort": { ... }
        "limit": 10
      }
    """
    out = dict(obj)

    # collection
    if "collection" not in out and "from" in out and isinstance(out["from"], str):
        out["collection"] = out.pop("from")
    if "collection" not in out and "coll" in out and isinstance(out["coll"], str):
        out["collection"] = out.pop("coll")

    # pipeline
    if "pipeline" not in out and "stages" in out and isinstance(out["stages"], list):
        out["pipeline"] = out.pop("stages")

    # operation inference
    if "operation" not in out:
        if "pipeline" in out and isinstance(out["pipeline"], list):
            out["operation"] = "aggregate"
        elif "filter" in out or "query" in out:
            out["operation"] = "find"
        else:
            out["operation"] = "aggregate" if "pipeline" in out else "unknown"

    # filter alias
    if "filter" not in out and "query" in out and isinstance(out["query"], dict):
        out["filter"] = out.pop("query")

    return out

def pipeline_stats(pipeline_obj: dict) -> dict:
    ops = []
    stages = pipeline_obj.get("pipeline")
    if isinstance(stages, list):
        for st in stages:
            if isinstance(st, dict) and len(st) == 1:
                ops.append(next(iter(st.keys())))
    return {
        "collection": pipeline_obj.get("collection"),
        "operation": pipeline_obj.get("operation"),
        "num_stages": len(stages) if isinstance(stages, list) else 0,
        "ops_used": ops[:20],
    }


# -----------------------------
# Prompting
# -----------------------------
SYSTEM_INSTRUCTIONS = """You are a precise assistant that converts a SQL intent into a MongoDB query object.
Output MUST be a single JSON object only (no prose, no markdown).
The JSON must be valid.

Target JSON schema (examples):
1) Aggregate:
{
  "collection": "<collection_name>",
  "operation": "aggregate",
  "pipeline": [ { "$match": {...}}, { "$group": {...}} ]
}

2) Find:
{
  "collection": "<collection_name>",
  "operation": "find",
  "filter": {...},
  "projection": {...},
  "sort": {...},
  "limit": 10
}

Rules:
- Use MongoDB operators correctly ($lookup, $unwind, $group, $project, $sort, $limit, $match).
- Prefer stable, executable pipelines.
- If a join is needed, use $lookup with correct localField/foreignField or let/pipeline form.
- Do not hallucinate tables/columns not present in provided schema/context.
"""

def build_prompt(
    question: str,
    db_id: str,
    retrieved_context: str = "",
    fewshot: str = "",
) -> str:
    parts = []
    parts.append(SYSTEM_INSTRUCTIONS)
    parts.append(f"DB_ID: {db_id}")
    if retrieved_context.strip():
        parts.append("RETRIEVED_CONTEXT:")
        parts.append(retrieved_context.strip())
    if fewshot.strip():
        parts.append("FEWSHOT_EXAMPLES:")
        parts.append(fewshot.strip())
    parts.append("QUESTION:")
    parts.append(question.strip())
    parts.append("OUTPUT_JSON:")
    return "\n\n".join(parts)


# -----------------------------
# Retrieval Hook
# -----------------------------
def run_retrieval(db_id: str, question: str, top_k: int = 6) -> Tuple[str, str]:
    """
    Returns (retrieved_context, fewshot_text)
    Uses rag_retrieve_fast_v2 if available; otherwise returns empty strings.
    """
    if not RAG_AVAILABLE or _rag_mod is None:
        return "", ""

    # Expect your module to expose some callable; we handle common patterns.
    # You can adapt these names to your real module functions.
    retrieved_context = ""
    fewshot_text = ""

    # Pattern A: retrieve_context(db_id, question, top_k)
    if hasattr(_rag_mod, "retrieve_context"):
        try:
            res = _rag_mod.retrieve_context(db_id=db_id, question=question, top_k=top_k)
            if isinstance(res, dict):
                retrieved_context = res.get("context", "") or res.get("retrieved_context", "") or ""
                fewshot_text = res.get("fewshot", "") or ""
            elif isinstance(res, tuple) and len(res) >= 1:
                retrieved_context = res[0] or ""
                if len(res) > 1:
                    fewshot_text = res[1] or ""
        except Exception:
            pass

    # Pattern B: rag(db_id, question)
    if not retrieved_context and hasattr(_rag_mod, "rag"):
        try:
            res = _rag_mod.rag(db_id, question)
            if isinstance(res, dict):
                retrieved_context = res.get("context", "") or ""
                fewshot_text = res.get("fewshot", "") or ""
        except Exception:
            pass

    return retrieved_context, fewshot_text


# -----------------------------
# Generation Backends
# -----------------------------
def ollama_generate(prompt: str, model: str, ollama_url: str, temperature: float = 0.0) -> str:
    """
    Uses Ollama /api/generate (not chat template), returns response text.
    """
    url = ollama_url.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": float(temperature),
        },
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    data = r.json()
    return data.get("response", "")

@dataclass
class MLXModelCache:
    model_dir: str
    model: Any = None
    tokenizer: Any = None

_MLX_CACHE: Optional[MLXModelCache] = None

def mlx_generate_text(prompt: str, model_dir: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    """
    Generates using MLX fused model folder.
    """
    global _MLX_CACHE

    if not MLX_AVAILABLE:
        raise RuntimeError("mlx-lm not available. Install with: pip install mlx-lm")

    if _MLX_CACHE is None or _MLX_CACHE.model_dir != model_dir or _MLX_CACHE.model is None:
        model, tokenizer = mlx_load(model_dir)
        _MLX_CACHE = MLXModelCache(model_dir=model_dir, model=model, tokenizer=tokenizer)

    return mlx_generate(
        _MLX_CACHE.model,
        _MLX_CACHE.tokenizer,
        prompt=prompt,
        max_tokens=int(max_tokens),
        temp=float(temperature),
    )


# -----------------------------
# Main pipeline
# -----------------------------
def generate_mongo(
    question: str,
    db_id: str,
    use_rag: bool,
    top_k: int,
    backend: str,
    ollama_url: str,
    ollama_model: str,
    mlx_model_dir: str,
    temperature: float,
    max_tokens: int,
) -> Tuple[str, str, str]:
    """
    Returns: (final_json_pretty, debug_text, raw_model_output)
    """
    t0 = now_ms()

    retrieved_context, fewshot = ("", "")
    if use_rag:
        retrieved_context, fewshot = run_retrieval(db_id=db_id, question=question, top_k=top_k)

    prompt = build_prompt(question=question, db_id=db_id, retrieved_context=retrieved_context, fewshot=fewshot)

    # Run model
    raw = ""
    if backend == "Ollama":
        raw = ollama_generate(prompt=prompt, model=ollama_model, ollama_url=ollama_url, temperature=temperature)
    elif backend == "MLX (LoRA fused)":
        raw = mlx_generate_text(prompt=prompt, model_dir=mlx_model_dir, max_tokens=max_tokens, temperature=temperature)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Parse + normalize
    obj = extract_first_json_obj(raw)
    debug = {
        "backend": backend,
        "db_id": db_id,
        "use_rag": use_rag,
        "top_k": top_k,
        "ollama_model": ollama_model if backend == "Ollama" else None,
        "mlx_model_dir": mlx_model_dir if backend.startswith("MLX") else None,
        "latency_ms": now_ms() - t0,
        "retrieved_context_chars": len(retrieved_context or ""),
        "fewshot_chars": len(fewshot or ""),
        "json_valid": bool(obj),
    }

    if not obj:
        # Return raw for inspection
        return (
            "",
            _dumps(debug) + "\n\n" + "ERROR: Could not parse valid JSON from model output.",
            raw.strip(),
        )

    norm = normalize_pipeline_obj(obj)
    stats = pipeline_stats(norm)
    debug["pipeline_stats"] = stats

    return (_dumps(norm), _dumps(debug), raw.strip())


# -----------------------------
# Gradio UI
# -----------------------------
def ui_generate(
    question: str,
    db_id: str,
    use_rag: bool,
    top_k: int,
    backend: str,
    ollama_url: str,
    ollama_model: str,
    mlx_model_dir: str,
    temperature: float,
    max_tokens: int,
):
    final_json, debug_text, raw = generate_mongo(
        question=question,
        db_id=db_id,
        use_rag=use_rag,
        top_k=top_k,
        backend=backend,
        ollama_url=ollama_url,
        ollama_model=ollama_model,
        mlx_model_dir=mlx_model_dir,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return final_json, raw, debug_text


def build_app() -> gr.Blocks:
    with gr.Blocks(title="SQL → Mongo (RAG + LoRA Demo)") as demo:
        gr.Markdown(
            "## Text-to-SQL → SQL-to-MongoDB (RAG + LoRA) Demo\n"
            "- Switch **Backend** to compare **Baseline (Ollama)** vs **LoRA fused (MLX)**.\n"
            "- Enable **RAG** if `rag_retrieve_fast_v2.py` exists and is configured.\n"
        )

        with gr.Row():
            question = gr.Textbox(
                label="User Question",
                placeholder="e.g., Find the top 5 customers by total purchase amount in 2023.",
                lines=3,
            )
        with gr.Row():
            db_id = gr.Textbox(label="DB ID", value=DEFAULT_DB_ID)
            backend = gr.Dropdown(
                choices=["Ollama", "MLX (LoRA fused)"],
                value="Ollama",
                label="Backend",
            )

        with gr.Accordion("Retrieval (RAG)", open=False):
            use_rag = gr.Checkbox(label=f"Enable RAG (available: {RAG_AVAILABLE})", value=RAG_AVAILABLE)
            top_k = gr.Slider(1, 12, value=6, step=1, label="Top-K Retrieved Items")

        with gr.Accordion("Ollama Settings", open=True):
            ollama_url = gr.Textbox(label="Ollama URL", value=DEFAULT_OLLAMA_URL)
            ollama_model = gr.Textbox(label="Ollama Model", value=DEFAULT_OLLAMA_MODEL)

        with gr.Accordion("MLX (LoRA fused) Settings", open=False):
            mlx_model_dir = gr.Textbox(
                label="MLX fused model directory",
                value=DEFAULT_MLX_MODEL_DIR,
                info="Path to fused model folder output by mlx_lm_lora.train (contains README.md).",
            )
            max_tokens = gr.Slider(128, 2048, value=768, step=64, label="Max New Tokens")

        with gr.Row():
            temperature = gr.Slider(0.0, 1.0, value=0.0, step=0.05, label="Temperature")
            run_btn = gr.Button("Generate Mongo JSON", variant="primary")

        with gr.Row():
            out_json = gr.Code(label="Final Mongo JSON (normalized)", language="json")
        with gr.Row():
            out_raw = gr.Textbox(label="Raw Model Output", lines=10)
        with gr.Row():
            out_debug = gr.Code(label="Debug / Stats", language="json")

        run_btn.click(
            fn=ui_generate,
            inputs=[
                question, db_id, use_rag, top_k, backend,
                ollama_url, ollama_model, mlx_model_dir,
                temperature, max_tokens
            ],
            outputs=[out_json, out_raw, out_debug],
        )

        gr.Markdown(
            "### Notes for your report/demo\n"
            "- **LoRA benefit** usually shows up as: fewer invalid JSONs, more consistent operator patterns, more stable `$lookup/$group` placement.\n"
            "- **Hybrid retrieval** (FTS + embeddings + few-shot) reduces hallucinations and improves schema alignment.\n"
        )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, show_error=True)

