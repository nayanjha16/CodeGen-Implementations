"""
Build SQL -> MongoDB dataset using Spider + rule-based converter.

Input:
    - Spider train_spider.json
Output:
    - JSON Lines file with:
        { "sql": <str>, "mongo": <list of stages>, "db_id": <str> }

Usage example:

    python -m src.sql2nosql.build_sql2mongo_dataset \
        --spider_path data/spider/train_spider.json \
        --output_path src/sql2nosql/data/sql2mongo_train.jsonl \
        --limit 10000
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from .sql_to_mongo_rule_based import SQLToMongoConverter


def build_dataset(
    spider_path: str,
    output_path: str,
    limit: int = -1,
) -> None:
    spider_file = Path(spider_path)
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with spider_file.open("r", encoding="utf-8") as f:
        spider_data = json.load(f)

    converter = SQLToMongoConverter()
    num_written = 0

    with out_file.open("w", encoding="utf-8") as out_f:
        for ex in spider_data:
            if limit > 0 and num_written >= limit:
                break

            sql = ex.get("query")
            db_id = ex.get("db_id")
            if not sql:
                continue

            try:
                mongo_pipeline = converter.convert(sql)
            except Exception as e:
                # Skip problematic queries for now
                # (You can log e + sql if you want)
                continue

            record: Dict[str, Any] = {
                "sql": sql,
                "mongo": mongo_pipeline,
                "db_id": db_id,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            num_written += 1

    print(f"Wrote {num_written} examples to {out_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SQL->Mongo dataset from Spider + rule-based converter.")
    parser.add_argument(
        "--spider_path",
        type=str,
        required=True,
        help="Path to Spider train_spider.json",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output JSONL path for SQL->Mongo pairs",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=-1,
        help="Maximum number of examples (<=0 means all).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_dataset(args.spider_path, args.output_path, args.limit)
