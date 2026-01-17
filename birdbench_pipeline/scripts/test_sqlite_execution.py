import os
import yaml
from datasets import load_from_disk

from pipelines.sqlite_executor import execute_sqlite_query

# --------------------------------------------------
# Load config
# --------------------------------------------------
with open("configs/paths.yaml") as f:
    paths = yaml.safe_load(f)

# --------------------------------------------------
# Load BirdBench Mini-Dev (HF metadata)
# --------------------------------------------------
dataset_path = os.path.join(
    paths["raw_birdbench"], "birdbench_mini_dev"
)

dataset = load_from_disk(dataset_path)
split = dataset["mini_dev_sqlite"]

# Pick one sample
sample = split[0] #Picking one sample query for testing

sql = sample["SQL"]
db_id = sample["db_id"]

# --------------------------------------------------
# Resolve SQLite DB path
# --------------------------------------------------
db_path = os.path.join(
    paths["birdbench_databases"],
    db_id,
    f"{db_id}.sqlite"
)

print("\n=== SQLITE EXECUTION TEST ===")
print("DB Path:", db_path)
print("\nSQL Query:\n", sql)

# --------------------------------------------------
# Execute SQL
# --------------------------------------------------
results = execute_sqlite_query(db_path, sql)

print("\n=== SQL RESULT ===")
if not results:
    print("(No rows returned)")
else:
    for row in results:
        print(row)
