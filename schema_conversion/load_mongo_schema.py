"""Load SQL data into MongoDB according to a converted schema.

This script either consumes a schema produced by `sql_to_mongo_schema.py` or
inspects the SQL database directly and writes documents into MongoDB using
PyMongo. The MongoDB database name will match the provided `--db-id` (or the
SQL connection string when not provided).

Usage (CLI):
  python schema_conversion/load_mongo_schema.py --sql-uri sqlite:///path/to/db.sqlite --mongo-uri mongodb://localhost:27017 --db-id mydb

NOTE: For safety, the script will not drop existing collections by default.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from pymongo import MongoClient
from sqlalchemy import MetaData, create_engine, select, Table

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def load_schema_from_file(path: str) -> Dict:
    p = Path(path)
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def fetch_table_rows(engine, table_name: str) -> List[Dict]:
    meta = MetaData()
    meta.reflect(bind=engine, only=[table_name])
    tbl = meta.tables[table_name]
    conn = engine.connect()
    stmt = select(tbl)
    rows = [dict(r) for r in conn.execute(stmt).mappings().all()]
    conn.close()
    return rows


def convert_and_insert(sql_uri: str, mongo_uri: str, db_id: Optional[str] = None, schema: Optional[Dict] = None) -> None:
    engine = create_engine(sql_uri)
    if schema is None:
        # analyze schema on the fly
        from .sql_to_mongo_schema import analyze_sql_schema

        schema = analyze_sql_schema(sql_uri)

    client = MongoClient(mongo_uri)
    if db_id is None:
        # derive database name from SQL uri - keep safe characters
        db_id = schema.get("db", "default_db").replace("/", "_").replace(":", "_")

    db = client[db_id]

    # Insert non-association tables first
    tables = schema.get("tables", {})
    embeds = schema.get("embeds", {})
    association_tables = set(schema.get("association_tables", []))

    inserted_collections = []

    for tname in tables:
        if tname in association_tables:
            logging.info("Skipping association table %s (many-to-many).", tname)
            continue

        rows = fetch_table_rows(engine, tname)
        # If there are child embeddings for this table, fetch child rows and group by FK
        embed_specs = embeds.get(tname, [])
        if embed_specs:
            # For each child, build a map parent_pk -> list(children)
            for spec in embed_specs:
                child = spec["child_table"]
                fk_col = spec["fk_column"]
                child_rows = fetch_table_rows(engine, child)
                grouped = defaultdict(list)
                for cr in child_rows:
                    parent_id = cr.get(fk_col)
                    grouped[parent_id].append(cr)

                # Merge into parent rows assuming first PK is parent id
                # Find parent PK column name
                parent_pk = tables[tname].get("pk", [])
                parent_pk_col = parent_pk[0] if parent_pk else None
                if parent_pk_col is None:
                    logging.warning("Cannot embed %s into %s because parent has no single PK.", child, tname)
                else:
                    for row in rows:
                        key = row.get(parent_pk_col)
                        row.setdefault(child, grouped.get(key, []))

        # Insert into MongoDB collection
        if rows:
            col = db[tname]
            col.insert_many(rows)
            inserted_collections.append(tname)
            logging.info("Inserted %d documents into %s.%s", len(rows), db_id, tname)
        else:
            logging.info("No rows found for table %s; skipping insert.", tname)

    logging.info("Finished. Collections inserted: %s", inserted_collections)


def main() -> None:
    parser = argparse.ArgumentParser(description="Load SQL data into MongoDB according to a converted schema")
    parser.add_argument("--sql-uri", required=True, help="SQLAlchemy URI to SQL db (e.g. sqlite:///...)")
    parser.add_argument("--mongo-uri", default="mongodb://localhost:27017", help="MongoDB connection URI")
    parser.add_argument("--db-id", default=None, help="MongoDB database name to create/use")
    parser.add_argument("--schema-file", default=None, help="Optional schema file created by sql_to_mongo_schema.py")
    args = parser.parse_args()

    schema = None
    if args.schema_file:
        schema = load_schema_from_file(args.schema_file)

    convert_and_insert(args.sql_uri, args.mongo_uri, db_id=args.db_id, schema=schema)


if __name__ == "__main__":
    main()
