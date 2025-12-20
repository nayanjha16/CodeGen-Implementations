# CodeGen-Implementation by Group 16 ✅
This project provides tools for converting natural language text to NoSQL (MongoDB) queries and schemas, and for validating conversions by executing equivalent SQL queries.

---

## Quickstart — Setup (venv) 🔧

1. Create and activate a virtual environment (recommended):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Upgrade pip and install dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

3. (Optional) If you plan to use Hugging Face datasets or model training, ensure you have `git-lfs` installed and are logged in to Hugging Face if required.

---

## Main scripts & usage ▶️

1. Prepare BIRD dataset (local folder, URL archive, or Hugging Face dataset):

```bash
# From a local directory
python scripts/prepare_bird.py --source-dir /path/to/bird_raw

# From a remote archive (zip/tar.gz)
python scripts/prepare_bird.py --download-url https://example.com/bird.zip

# Directly from Hugging Face dataset (example)
python scripts/prepare_bird.py --hf-dataset sisinflab-ai/GradeSQL-training-dataset-bird-unbalanced
```

This writes raw files to `data/bird/` (by default) and produces a filtered JSONL at `data/bird_filtered/filtered.jsonl`.

2. Extract & filter SQL queries (if you already have `data/bird/`):

```bash
python schema_conversion/extract_schema.py --input-dir data/bird --output-dir data/bird_filtered
```

3. Analyze SQL schema and produce Mongo-style schema hints:

```bash
python schema_conversion/sql_to_mongo_schema.py --sql-uri sqlite:///path/to/db.sqlite --out-file data/bird_filtered/mongo_schema.json
```

4. Load converted schema/data into MongoDB (uses PyMongo):

```bash
python schema_conversion/load_mongo_schema.py --sql-uri sqlite:///path/to/db.sqlite --mongo-uri mongodb://localhost:27017 --db-id mydb
```

5. Convert SQL queries to Mongo aggregation pipelines and verify execution:

```bash
python query_conversion/verify_execution.py --input-file data/bird_filtered/filtered.jsonl --sql-uri sqlite:///path/to/db.sqlite --mongo-uri mongodb://localhost:27017 --out-dir data/gold_queries
```

Matched queries (where SQL and Mongo outputs match) are written to `data/gold_queries/matched.jsonl`.

---

## Model training & inference (schema & query generators) 🧠

Note: Training large LLMs requires GPUs and non-trivial time/resources. The repo includes training scripts that use PEFT/LoRA for efficient fine-tuning.

1. Schema predictor training (prepare JSONL with `nlq`, `schema`, optional `gold_schema`):

```bash
python models/schema_predictor/train.py --train-file path/to/train.jsonl --model-name <hf-model> --output-dir outputs/schema_predictor --epochs 1 --per-device-batch-size 4
```

2. Schema predictor inference:

```bash
python models/schema_predictor/infer.py --model-dir outputs/schema_predictor --nlq "Find users and totals" --schema-file some_schema.json
```

3. Query generator training (prepare JSONL with `nlq`, `pred_schema`, `gold_pipeline`):

```bash
python models/query_generator/train.py --train-file path/to/train.jsonl --model-name <hf-model> --output-dir outputs/query_generator --epochs 1
```

4. Query generator inference (generate pipeline JSON array):

```bash
python models/query_generator/infer.py --nlq "Total order amount per user" --schema-file pred_schema.json
```

The inference code includes heuristic fallbacks so you can test without training a model.

---

## Smoke tests & quick checks ✅

Run included smoke tests to validate logic without external services:

```bash
# Extractor smoke test
python schema_conversion/_smoke_test.py

# Schema->Mongo smoke test
python schema_conversion/_smoke_sql_to_mongo.py

# Query conversion smoke test (in-memory verification)
python query_conversion/_smoke_verify.py

# Schema predictor inference smoke
python models/schema_predictor/_smoke_infer.py

# Query generator inference smoke
python models/query_generator/_smoke_infer.py
```

---

## Notes & troubleshooting ⚠️

- Ensure required services are running when using MongoDB commands (`mongod` / a remote cluster). The scripts will attempt in-memory fallbacks when possible.
- If a Hugging Face dataset requires authentication, run `huggingface-cli login` before invoking `scripts/prepare_bird.py --hf-dataset ...`.
- For large models use mixed precision (`--fp16`) and consider using multi-GPU or accelerating libraries (bitsandbytes).

---

If you'd like, I can add a `Makefile` or `tasks.json` for VS Code to make these commands one-liners. Want me to add that? 🔧


