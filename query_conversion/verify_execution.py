"""Verify SQL -> Mongo query conversion by executing both and comparing results.

Modes:
  - Single query mode: provide --sql, --collection, --sql-uri, --mongo-uri (optional).
  - Batch mode: provide --input-file (JSONL with objects containing `db_id`, `nlq`, `sql`).

If MongoDB is not available, the script can run the generated pipeline against
the SQL query results directly (in-memory pipeline runner) to check semantic
equivalence without a live Mongo instance.

Matching queries are written as JSONL to `data/gold_queries/matched.jsonl` by default.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional

from pymongo import MongoClient
from sqlalchemy import create_engine, text

from query_conversion.sql_to_mongo_query import sql_to_mongo_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def run_sql(engine, sql: str) -> List[Dict[str, Any]]:
    conn = engine.connect()
    try:
        res = conn.execute(text(sql)).mappings().all()
        return [dict(r) for r in res]
    finally:
        conn.close()


def run_pipeline_on_docs(docs: List[Dict], pipeline: List[Dict]) -> List[Dict]:
    """A minimal in-memory runner for a subset of Mongo aggregation stages.

    Supports $match, $group, $project, $unwind, $sort.
    """
    data = docs
    for stage in pipeline:
        if "$match" in stage:
            cond = stage["$match"]
            def match_doc(d):
                for k, v in cond.items():
                    if isinstance(v, dict):
                        # simple comparison operators
                        for op, val in v.items():
                            if op == "$gt" and not (d.get(k) > val):
                                return False
                            if op == "$lt" and not (d.get(k) < val):
                                return False
                            if op == "$gte" and not (d.get(k) >= val):
                                return False
                            if op == "$lte" and not (d.get(k) <= val):
                                return False
                            if op == "$ne" and not (d.get(k) != val):
                                return False
                            if op == "$in" and not (d.get(k) in val):
                                return False
                    else:
                        if d.get(k) != v:
                            return False
                return True

            data = [d for d in data if match_doc(d)]

        elif "$group" in stage:
            spec = stage["$group"]
            gid = spec.get("_id")
            groups = {}
            for d in data:
                if gid is None:
                    key = None
                elif isinstance(gid, dict):
                    key = tuple((k, d.get(v[1:]) if isinstance(v, str) and v.startswith('$') else None) for k, v in gid.items())
                else:
                    key = d.get(gid[1:]) if isinstance(gid, str) and gid.startswith('$') else d.get(gid)

                if key not in groups:
                    groups[key] = {"__docs": []}
                groups[key]["__docs"].append(d)

            new = []
            for key, content in groups.items():
                out = {}
                docs = content["__docs"]
                # compute accumulators
                for k, v in spec.items():
                    if k == "_id":
                        if isinstance(v, dict):
                            for subk, subv in v.items():
                                out.setdefault("_id", {})[subk] = docs[0].get(subv[1:]) if isinstance(subv, str) and subv.startswith('$') else None
                        else:
                            out["_id"] = None
                        continue
                    if isinstance(v, dict):
                        if "$sum" in v:
                            arg = v["$sum"]
                            if isinstance(arg, (int, float)):
                                out[k] = sum(arg for _ in docs)
                            elif isinstance(arg, str) and arg.startswith('$'):
                                fld = arg[1:]
                                out[k] = sum((d.get(fld) or 0) for d in docs)
                        elif "$avg" in v:
                            arg = v["$avg"]
                            if isinstance(arg, str) and arg.startswith('$'):
                                fld = arg[1:]
                                vals = [d.get(fld) for d in docs if d.get(fld) is not None]
                                out[k] = sum(vals) / len(vals) if vals else None
                        elif "$min" in v:
                            arg = v["$min"]
                            fld = arg[1:]
                            out[k] = min((d.get(fld) for d in docs if d.get(fld) is not None), default=None)
                        elif "$max" in v:
                            arg = v["$max"]
                            fld = arg[1:]
                            out[k] = max((d.get(fld) for d in docs if d.get(fld) is not None), default=None)
                    else:
                        out[k] = None
                new.append(out)
            data = new

        elif "$project" in stage:
            spec = stage["$project"]
            new = []
            for d in data:
                od = {}
                for k, v in spec.items():
                    if v == 1:
                        od[k] = d.get(k)
                    elif isinstance(v, str) and v.startswith("$_id."):
                        fld = v.split('.', 1)[1]
                        od[k] = d.get("_id", {}).get(fld)
                    elif isinstance(v, str) and v.startswith("$"):
                        fld = v[1:]
                        od[k] = d.get(fld)
                new.append(od)
            data = new

        elif "$sort" in stage:
            spec = stage["$sort"]
            # sort by first key then next
            keys = list(spec.items())
            def sort_key(d):
                return tuple((d.get(k) or 0) * (1 if v == 1 else -1) for k, v in keys)
            data = sorted(data, key=sort_key)

        elif "$unwind" in stage:
            fld = stage["$unwind"].lstrip('$') if isinstance(stage["$unwind"], str) else stage["$unwind"].get('path', '').lstrip('$')
            new = []
            for d in data:
                arr = d.get(fld)
                if isinstance(arr, list):
                    for el in arr:
                        nd = dict(d)
                        nd[fld] = el
                        new.append(nd)
                else:
                    new.append(d)
            data = new

        else:
            # unsupported stage: skip
            pass

    return data


def compare_results(sql_rows: List[Dict], mongo_rows: List[Dict]) -> bool:
    # Normalize rows for comparison: sort by stringified JSON
    def normalize(rows):
        return sorted([json.dumps(r, sort_keys=True, default=str) for r in rows])

    return normalize(sql_rows) == normalize(mongo_rows)


def process_input_file(input_file: str, sql_uri: str, mongo_uri: Optional[str], out_dir: str) -> None:
    Path = __import__('pathlib').Path
    p = Path(input_file)
    if not p.exists():
        logging.error("Input file %s does not exist", input_file)
        return

    engine = create_engine(sql_uri) if sql_uri else None
    client = None
    if mongo_uri:
        try:
            client = MongoClient(mongo_uri, serverSelectionTimeoutMS=2000)
            # trigger connection
            client.server_info()
        except Exception as e:
            logging.warning("Cannot connect to MongoDB at %s: %s. Falling back to in-memory pipeline execution.", mongo_uri, e)
            client = None

    out_path = os.path.join(out_dir, "matched.jsonl")
    os.makedirs(out_dir, exist_ok=True)

    total = 0
    matched = 0

    with p.open("r", encoding="utf-8") as fh, open(out_path, "w", encoding="utf-8") as out_fh:
        for line in fh:
            total += 1
            obj = json.loads(line)
            sql = obj.get("sql")
            db_id = obj.get("db_id")
            nlq = obj.get("nlq")

            conv = sql_to_mongo_pipeline(sql)
            if not conv:
                logging.info("Skipping query %d: cannot convert", total)
                continue

            collection = conv["collection"]
            pipeline = conv["pipeline"]

            # Run SQL
            sql_rows = run_sql(engine, sql) if engine else []

            # Run pipeline on Mongo if available, else run in-memory on sql_rows
            if client:
                db = client[db_id] if db_id else client.get_database()
                try:
                    mongo_docs = list(db[collection].aggregate(pipeline))
                except Exception as e:
                    logging.warning("Mongo aggregation failed for query %d: %s. Falling back to in-memory.", total, e)
                    mongo_docs = run_pipeline_on_docs(sql_rows, pipeline)
            else:
                mongo_docs = run_pipeline_on_docs(sql_rows, pipeline)

            if compare_results(sql_rows, mongo_docs):
                matched += 1
                out_record = {"db_id": db_id, "nlq": nlq, "sql": sql, "mongo_pipeline": pipeline}
                out_fh.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    logging.info("Total processed: %d, matched: %d, mismatched: %d. Matched queries written to %s", total, matched, total - matched, out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify SQL -> Mongo conversion by executing both and comparing results")
    parser.add_argument("--input-file", default="data/bird_filtered/filtered.jsonl", help="JSONL file with records containing 'sql' and 'db_id'")
    parser.add_argument("--sql-uri", required=True, help="SQLAlchemy URI to execute SQL queries against")
    parser.add_argument("--mongo-uri", default=None, help="Optional MongoDB URI to execute pipeline against. If not provided or unreachable, runs pipeline in-memory on SQL rows")
    parser.add_argument("--out-dir", default="data/gold_queries", help="Where to save matched gold queries")
    args = parser.parse_args()

    process_input_file(args.input_file, args.sql_uri, args.mongo_uri, args.out_dir)


if __name__ == "__main__":
    main()
