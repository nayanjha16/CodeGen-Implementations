import os
import yaml
from datasets import load_from_disk

from pipelines.sql_to_mongo import sql_to_mongo_pipeline
from pipelines.mongo_executor import execute_mongo_query
from pipelines.sqlite_executor import execute_sqlite_query
from evaluation.result_normalizer import normalize_results

# --------------------------------------------------
# Load config
# --------------------------------------------------
with open("configs/paths.yaml") as f:
    paths = yaml.safe_load(f)

# --------------------------------------------------
# Load BirdBench sample
# --------------------------------------------------
dataset_path = os.path.join(
    paths["raw_birdbench"], "birdbench_mini_dev"
)
dataset = load_from_disk(dataset_path)
sample = dataset["mini_dev_sqlite"][0]

sql = sample["SQL"]
db_id = sample["db_id"]

sqlite_db_path = os.path.join(
    paths["birdbench_databases"],
    db_id,
    f"{db_id}.sqlite"
)

# --------------------------------------------------
# Run SQL
# --------------------------------------------------
sql_result = execute_sqlite_query(sqlite_db_path, sql)
sql_result = normalize_results(sql_result)

# --------------------------------------------------
# Generate Mongo pipeline using LLM
# --------------------------------------------------
pipeline = sql_to_mongo_pipeline(
    sql=sql,
    sqlite_db_path=sqlite_db_path,
    prompt_template_path="llm/prompts/sql_to_mongo.txt"
)

print("\n=== GENERATED MONGO PIPELINE ===")
print(pipeline)

# --------------------------------------------------
# Execute Mongo query
# --------------------------------------------------
mongo_result = execute_mongo_query(
    mongo_db_name=db_id,
    collection_name=list(pipeline[0].values())[0] if False else "customers",
    pipeline=pipeline
)
mongo_result = normalize_results(mongo_result)

print("\n=== SQL RESULT ===")
print(sql_result)

print("\n=== MONGO RESULT ===")
print(mongo_result)

print("\n=== MATCH ===")
print(sql_result == mongo_result)
