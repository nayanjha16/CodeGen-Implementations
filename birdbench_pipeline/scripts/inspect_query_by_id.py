import os
import yaml
from datasets import load_from_disk
from pipelines.sqlite_executor import execute_sqlite_query

QUESTION_IDS = {1473, 1476, 1479, 1480, 1482, 1483, 1490, 1493, 1498, 1505}

with open("configs/paths.yaml") as f:
    paths = yaml.safe_load(f)

dataset = load_from_disk(
    os.path.join(paths["raw_birdbench"], "birdbench_mini_dev")
)

split = dataset["mini_dev_sqlite"]

for sample in split:
    if sample["question_id"] in QUESTION_IDS:
        qid = sample["question_id"]
        sql = sample["SQL"]
        db_id = sample["db_id"]

        db_path = os.path.join(
            paths["birdbench_databases"],
            db_id,
            f"{db_id}.sqlite"
        )

        print("\n" + "="*80)
        print(f"QUESTION ID: {qid}")
        print("DB:", db_id)
        print("\nSQL:")
        print(sql)

        try:
            result = execute_sqlite_query(db_path, sql)
            print("\nSQL RESULT:")
            print(result)
        except Exception as e:
            print("\nSQL EXECUTION ERROR:", e)
