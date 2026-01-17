import os
import yaml
import json
from datasets import load_from_disk
from tqdm import tqdm

from pipelines.sqlite_executor import execute_sqlite_query
from pipelines.sql_to_mongo import sql_to_mongo_pipeline
from pipelines.execution_loop import run_with_correction
from evaluation.result_normalizer import normalize_results

# --------------------------------------------------
# CONFIG
# --------------------------------------------------
MAX_QUERIES = 20
SAVE_EVERY = 10

# --------------------------------------------------
# Load paths
# --------------------------------------------------
with open("configs/paths.yaml") as f:
    paths = yaml.safe_load(f)

# --------------------------------------------------
# Load BirdBench
# --------------------------------------------------
dataset = load_from_disk(
    os.path.join(paths["raw_birdbench"], "birdbench_mini_dev")
)
split = dataset["mini_dev_sqlite"]
num_samples = min(MAX_QUERIES, len(split))

results_log = []
success_count = 0
retry_sum = 0

os.makedirs("logs", exist_ok=True)
checkpoint_path = "logs/step8_results.json"

# --------------------------------------------------
# MAIN LOOP
# --------------------------------------------------
for idx in tqdm(range(num_samples), desc="Running Step 8"):
    sample = split[idx]
    sql = sample["SQL"]
    db_id = sample["db_id"]
    qid = sample.get("question_id", idx)

    sqlite_db_path = os.path.join(
        paths["birdbench_databases"],
        db_id,
        f"{db_id}.sqlite"
    )

    try:
        # ---------------------------
        # SQL execution
        # ---------------------------
        sql_result = execute_sqlite_query(sqlite_db_path, sql)
        sql_result = normalize_results(sql_result)

        # ---------------------------
        # LLM SQL → Mongo
        # ---------------------------
        mongo_pipeline, root_collection = sql_to_mongo_pipeline(
            sql=sql,
            sqlite_db_path=sqlite_db_path,
            prompt_template_path="llm/prompts/sql_to_mongo_rag.txt"
        )

        # 🔴 STORE INITIAL PIPELINE
        initial_mongo_pipeline = mongo_pipeline

        # ---------------------------
        # Auto-correction loop
        # ---------------------------
        final_pipeline, success = run_with_correction(
            sql=sql,
            sql_result=sql_result,
            mongo_db=db_id,
            collection=root_collection,
            initial_pipeline=mongo_pipeline,
            fix_prompt_path="llm/prompts/sql_to_mongo_fewshot.txt" #"llm/prompts/sql_to_mongo_fix.txt"
        )

        retries = 0 if success else 1
        success_count += int(success)
        retry_sum += retries

        # 🔴 FULL LOG ENTRY
        results_log.append({
            "question_id": qid,
            "db_id": db_id,
            "birdbench_sql": sql,
            "root_collection": root_collection,
            "initial_mongo_pipeline": initial_mongo_pipeline,
            "final_mongo_pipeline": final_pipeline,
            "success": success,
            "retries": retries
        })

    except Exception as e:
        results_log.append({
            "question_id": qid,
            "db_id": db_id,
            "birdbench_sql": sql,
            "success": False,
            "error": str(e)
        })

    # ---------------------------
    # Save checkpoints
    # ---------------------------
    if (idx + 1) % SAVE_EVERY == 0:
        with open(checkpoint_path, "w") as f:
            json.dump(results_log, f, indent=2)

# --------------------------------------------------
# FINAL SUMMARY
# --------------------------------------------------
total = len(results_log)
accuracy = success_count / total

summary = {
    "total_queries": total,
    "success_count": success_count,
    "execution_accuracy": accuracy,
    "avg_retries": retry_sum / max(success_count, 1)
}

print("\n=== STEP 8 SUMMARY ===")
print(json.dumps(summary, indent=2))

with open("logs/step8_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
