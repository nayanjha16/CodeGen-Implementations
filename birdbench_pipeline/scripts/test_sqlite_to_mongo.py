import os
import yaml
from datasets import load_from_disk

from db.converters.sqlite_to_mongo import sqlite_to_mongo

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

db_id = sample["db_id"]

# --------------------------------------------------
# Resolve SQLite DB path
# --------------------------------------------------
sqlite_db_path = os.path.join(
    paths["birdbench_databases"],
    db_id,
    f"{db_id}.sqlite"
)

# --------------------------------------------------
# Convert SQLite → MongoDB
# --------------------------------------------------
sqlite_to_mongo(
    sqlite_db_path=sqlite_db_path,
    mongo_db_name=db_id
)
