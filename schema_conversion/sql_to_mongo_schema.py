"""Convert SQL schema (via SQLAlchemy) into MongoDB-style JSON schema.

This module inspects an existing SQL database using SQLAlchemy reflection,
detects foreign key relationships and applies heuristic rules to decide
whether to embed (one-to-many / lookup tables) or flatten/skip (many-to-many).

Usage (CLI):
  python schema_conversion/sql_to_mongo_schema.py --sql-uri sqlite:///path/to/db.sqlite --out-file data/bird_filtered/mongo_schema.json

Expose functions for programmatic use:
  - analyze_sql_schema(sql_uri) -> dict
  - save_schema(schema, out_file)
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from sqlalchemy import MetaData, create_engine, Table

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


@dataclass
class TableInfo:
    name: str
    columns: List[str]
    pk: List[str]
    fks: List[Tuple[str, str]]  # list of (column_name, referred_table)


def analyze_sql_schema(sql_uri: str) -> Dict:
    """Reflect the SQL schema and return a JSON-serializable schema dict.

    The returned dict has keys:
      - db: derived from connection (or provided uri)
      - tables: mapping table_name -> TableInfo-like dict
      - relationships: embeddings / many-to-many decisions
    """
    engine = create_engine(sql_uri)
    meta = MetaData()
    meta.reflect(bind=engine)

    tables: Dict[str, TableInfo] = {}

    # Build basic table info
    for tbl in meta.sorted_tables:
        cols = [c.name for c in tbl.columns]
        pk = [c.name for c in tbl.primary_key]
        fks = []
        for c in tbl.columns:
            for fk in c.foreign_keys:
                # fk.column.table.name is the parent table
                fks.append((c.name, fk.column.table.name))

        tables[tbl.name] = TableInfo(name=tbl.name, columns=cols, pk=pk, fks=fks)

    # Detect lookup tables (heuristic: small number of columns and no foreign keys)
    lookup_tables: Set[str] = set()
    for tname, info in tables.items():
        if len(info.columns) <= 2 and not info.fks:
            lookup_tables.add(tname)

    # Detect association (many-to-many) tables: only foreign keys and typically 2 FKs
    association_tables: Set[str] = set()
    for tname, info in tables.items():
        if info.fks and len(info.fks) >= 2 and len(info.columns) <= len(info.fks) + 1:
            # heuristic match
            association_tables.add(tname)

    # Build relationships: for each FK from child->parent, mark child as candidate for embedding into parent
    embeds: Dict[str, List[Dict]] = defaultdict(list)
    many_to_many: List[Dict] = []

    for tname, info in tables.items():
        # skip association tables
        if tname in association_tables:
            many_to_many.append({"association_table": tname, "fks": info.fks})
            continue

        for col, parent in info.fks:
            # If child table is simple (only PK + FK) or is identified as lookup, embed into parent
            child_info = info
            is_lookup = tname in lookup_tables
            simple_child = len(child_info.columns) <= 3
            if is_lookup or simple_child:
                embeds[parent].append({"child_table": tname, "fk_column": col, "child_columns": child_info.columns})
            else:
                # otherwise, treat as separate collection; can be joined but not embedded
                pass

    schema = {
        "db": sql_uri,
        "tables": {t: {"columns": info.columns, "pk": info.pk, "fks": info.fks} for t, info in tables.items()},
        "lookup_tables": sorted(list(lookup_tables)),
        "association_tables": sorted(list(association_tables)),
        "embeds": embeds,
        "many_to_many": many_to_many,
    }

    return schema


def save_schema(schema: Dict, out_file: str) -> None:
    p = Path(out_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(schema, fh, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze SQL schema and produce Mongo-style JSON schema hints")
    parser.add_argument("--sql-uri", required=True, help="SQLAlchemy uri to the SQL database (e.g. sqlite:///path/db.sqlite)")
    parser.add_argument("--out-file", required=False, default="data/bird_filtered/mongo_schema.json", help="Where to write JSON schema file")
    args = parser.parse_args()

    schema = analyze_sql_schema(args.sql_uri)
    save_schema(schema, args.out_file)
    logging.info("Schema written to %s", args.out_file)


if __name__ == "__main__":
    main()
