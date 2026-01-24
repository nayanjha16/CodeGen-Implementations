#!/usr/bin/env bash
set -euo pipefail

# -------- USER CONFIG --------
TRAIN_DIR="/Users/pavanpratyusha/Desktop/train"
OUT_DIR="/Users/pavanpratyusha/Desktop/bird_outputs"
MODEL="qwen2.5"
OLLAMA_URL="http://127.0.0.1:11434"
MONGO_URI="mongodb://localhost:27017"
START="${1:-100}"
END="${2:-200}"

# RAG index path
RAG_INDEX="$OUT_DIR/rag_index.sqlite"

# -------- STEP 0: venv (assumes already active) --------
mkdir -p "$OUT_DIR"

echo "[STEP 1] Build/refresh RAG index (schema + rules + embeddings)"
python -u rag_index_build.py \
  --train-tables "$TRAIN_DIR/train_tables.json" \
  --out-index "$RAG_INDEX" \
  --with-embeddings \
  --reset

echo "[STEP 2] Generate NoSQL JSONL (resume-safe)"

NOSQL_JSONL="$OUT_DIR/nosql_queries_${START}_${END}.jsonl"
REPORT_JSONL="$OUT_DIR/exec_report_${START}_${END}.jsonl"
ERRORS_JSONL="$OUT_DIR/exec_errors_${START}_${END}.jsonl"

echo "[STEP 3] Verify execution accuracy (resume-safe)"
python -u verify_exec_accuracy.py \
  --nosql-jsonl "$NOSQL_JSONL" \
  --sqlite-root "$TRAIN_DIR/train_databases" \
  --mongo-uri "$MONGO_URI" \
  --out-report "$REPORT_JSONL" \
  --out-errors "$ERRORS_JSONL" \
  --resume

echo "[DONE] Report: $REPORT_JSONL"
