from datasets import load_from_disk
import yaml
import os
import pprint

# --------------------------------------------------
# Load paths configuration
# --------------------------------------------------
with open("configs/paths.yaml") as f:
    paths = yaml.safe_load(f)

# --------------------------------------------------
# Load BirdBench Mini-Dev dataset (HF metadata only)
# --------------------------------------------------
dataset_path = os.path.join(
    paths["raw_birdbench"],
    "birdbench_mini_dev"
)

dataset = load_from_disk(dataset_path)

print("Available splits:", list(dataset.keys()))

# Always use SQLite split for our pipeline
split_name = "mini_dev_sqlite"
split = dataset[split_name]

print(f"\nUsing split: {split_name}")
print(f"Number of examples: {len(split)}")

# --------------------------------------------------
# Inspect a sample
# --------------------------------------------------
sample_idx = 0
sample = split[sample_idx]

print(f"\n=== SAMPLE RECORD (index={sample_idx}) ===")
pprint.pprint(sample)

# --------------------------------------------------
# Extract canonical BirdBench fields
# --------------------------------------------------
question = sample["question"]
sql = sample["SQL"]
db_id = sample["db_id"]

print("\n=== EXTRACTED FIELDS ===")
print("Question :", question)
print("SQL      :", sql)
print("DB ID    :", db_id)

# --------------------------------------------------
# Resolve SQLite DB path (ACTUAL database)
# --------------------------------------------------
sqlite_db_path = os.path.join(
    paths["birdbench_databases"],
    db_id,
    f"{db_id}.sqlite"
)

print("\n=== SQLITE DATABASE RESOLUTION ===")
print("Resolved SQLite DB Path:", sqlite_db_path)
print("DB Exists:", os.path.exists(sqlite_db_path))
