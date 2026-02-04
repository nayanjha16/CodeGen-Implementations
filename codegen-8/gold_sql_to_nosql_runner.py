#!/usr/bin/env python3
"""
gold_sql_to_nosql_runner.py

Goal
----
Process ONLY GOLD queries (train_gold.sql) aligned with train.json, and output
a single JSONL file ready for SQL->NoSQL conversion/evaluation pipelines.

Inputs (BIRD / Spider-style)
----------------------------
- train.json: list of items with fields: question_id, db_id, question, sql (sometimes present)
- train_gold.sql: gold SQL, line-aligned with train.json
- train_tables.json: schema metadata (used for optional Algorithm-1 DB transformation)

Outputs
-------
1) nosql_queries_gold.jsonl (always)
   One JSON object per gold query:
     {
       "question_id": ...,
       "db_id": ...,
       "question": ...,
       "sql_gold": "...",
       "mongo": { "collection": "...", "operation": "...", "pipeline": [...] },
       "meta": {...}
     }

2) Optional: all_dbs_nested.jsonl (ONLY if --sqlite-root and --out-nested-jsonl provided)
   Nested documents created by Algorithm-1 (DB transformation) from arXiv:2502.11201

Usage
-----
python gold_sql_to_nosql_runner.py \
  --train-json /Users/pavanpratyusha/Desktop/train/train.json \
  --train-gold-sql /Users/pavanpratyusha/Desktop/train/train_gold.sql \
  --train-tables /Users/pavanpratyusha/Desktop/train/train_tables.json \
  --out-jsonl /Users/pavanpratyusha/Desktop/bird_outputs/nosql_queries_gold.jsonl

Optional nested export (Algorithm-1):
python gold_sql_to_nosql_runner.py \
  ...same as above... \
  --sqlite-root /Users/pavanpratyusha/Desktop/train/train_databases \
  --out-nested-jsonl /Users/pavanpratyusha/Desktop/bird_outputs/all_dbs_nested.jsonl \
  --max-children 50
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Set, Optional, Iterable


# ============================================================
# Part A) GOLD QUERY DRIVER (train_gold.sql aligned with train.json)
# ============================================================

def load_train_json(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("train.json must be a JSON list")
    return data


def load_gold_sql_lines(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]
    # Keep even empty lines (rare) to preserve alignment
    return lines


def sql_to_mongo_placeholder(sql: str, db_id: str) -> Dict[str, Any]:
    """
    Placeholder "conversion" to keep output schema consistent.
    Replace this with your real SQL->Mongo converter (LLM/RAG/rules).

    For now:
      - operation is 'aggregate'
      - pipeline contains a $comment with the SQL
      - collection defaults to db_id (or you can set later after DB transform)
    """
    return {
        "collection": db_id,              # you may later replace with root collection name
        "operation": "aggregate",
        "pipeline": [
            {"$comment": {"sql_gold": sql}}
        ]
    }


def build_gold_jsonl(train_items: List[Dict[str, Any]], gold_sql: List[str], out_jsonl: str) -> None:
    if len(train_items) != len(gold_sql):
        raise ValueError(
            f"Alignment mismatch: train.json items={len(train_items)} "
            f"but train_gold.sql lines={len(gold_sql)}"
        )

    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)

    with open(out_jsonl, "w", encoding="utf-8") as out_f:
        for i, item in enumerate(train_items):
            qid = item.get("question_id", i)
            db_id = item.get("db_id")
            question = item.get("question")
            sql_gold = gold_sql[i]

            rec = {
                "question_id": qid,
                "db_id": db_id,
                "question": question,
                "sql_gold": sql_gold,
                "mongo": sql_to_mongo_placeholder(sql_gold, db_id),
                "meta": {
                    "source": "train_gold.sql",
                    "aligned_index": i
                }
            }
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")


# ============================================================
# Part B) OPTIONAL: Algorithm-1 DB Transformation (Nested NoSQL docs)
#         (arXiv:2502.11201) - for your DB conversion stage
# ============================================================

@dataclass(frozen=True)
class FK:
    src_col: int
    dst_col: int
    src_table: int
    dst_table: int
    src_col_name: str
    dst_col_name: str


@dataclass
class Schema:
    db_id: str
    table_names: List[str]
    column_names: List[Tuple[int, str]]
    column_types: List[str]
    primary_keys: List[int]
    foreign_keys: List[FK]


def load_schemas(train_tables_path: str) -> Dict[str, Schema]:
    with open(train_tables_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    out: Dict[str, Schema] = {}
    for item in raw:
        db_id = item["db_id"]
        table_names = item["table_names_original"]
        column_names = [tuple(x) for x in item["column_names_original"]]
        column_types = item["column_types"]
        primary_keys = item["primary_keys"]

        fks: List[FK] = []
        for src_col, dst_col in item.get("foreign_keys", []):
            src_t, src_name = column_names[src_col]
            dst_t, dst_name = column_names[dst_col]
            fks.append(FK(
                src_col=src_col,
                dst_col=dst_col,
                src_table=src_t,
                dst_table=dst_t,
                src_col_name=src_name,
                dst_col_name=dst_name
            ))

        out[db_id] = Schema(
            db_id=db_id,
            table_names=table_names,
            column_names=column_names,
            column_types=column_types,
            primary_keys=primary_keys,
            foreign_keys=fks
        )
    return out


def build_table_graph(schema: Schema) -> Dict[int, Set[int]]:
    g: Dict[int, Set[int]] = defaultdict(set)
    for fk in schema.foreign_keys:
        if fk.src_table < 0 or fk.dst_table < 0:
            continue
        g[fk.src_table].add(fk.dst_table)
        g[fk.dst_table].add(fk.src_table)
    for t in range(len(schema.table_names)):
        g[t] = g.get(t, set())
    return g


def connected_components(graph: Dict[int, Set[int]]) -> List[Set[int]]:
    seen: Set[int] = set()
    comps: List[Set[int]] = []
    for start in graph.keys():
        if start in seen:
            continue
        q = deque([start])
        comp = set()
        seen.add(start)
        while q:
            u = q.popleft()
            comp.add(u)
            for v in graph[u]:
                if v not in seen:
                    seen.add(v)
                    q.append(v)
        comps.append(comp)
    return comps


def detect_fk_cycles(schema: Schema, comp: Set[int]) -> bool:
    adj = defaultdict(set)
    for fk in schema.foreign_keys:
        if fk.src_table in comp and fk.dst_table in comp:
            adj[fk.src_table].add(fk.dst_table)

    WHITE, GRAY, BLACK = 0, 1, 2
    color = {t: WHITE for t in comp}

    def dfs(u: int) -> bool:
        color[u] = GRAY
        for v in adj[u]:
            if color[v] == GRAY:
                return True
            if color[v] == WHITE and dfs(v):
                return True
        color[u] = BLACK
        return False

    for t in comp:
        if color[t] == WHITE and dfs(t):
            return True
    return False


def choose_main_table(schema: Schema, comp: Set[int]) -> int:
    indeg = defaultdict(int)
    outdeg = defaultdict(int)
    for fk in schema.foreign_keys:
        if fk.src_table in comp and fk.dst_table in comp:
            outdeg[fk.src_table] += 1
            indeg[fk.dst_table] += 1

    best_t = None
    best_key = None
    for t in comp:
        key = (indeg[t], -outdeg[t], -t)
        if best_t is None or key > best_key:
            best_t = t
            best_key = key
    assert best_t is not None
    return best_t


def sqlite_table_exists(conn: sqlite3.Connection, table: str) -> bool:
    cur = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table,),
    )
    return cur.fetchone() is not None


def sqlite_fetch_all(conn: sqlite3.Connection, table: str) -> List[Dict[str, Any]]:
    cur = conn.execute(f'SELECT * FROM "{table}"')
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def get_children_fks(schema: Schema, parent_table_idx: int) -> List[FK]:
    return [fk for fk in schema.foreign_keys if fk.dst_table == parent_table_idx]


def embed_children(
    schema: Schema,
    conn: sqlite3.Connection,
    parent_table_idx: int,
    parent_rows: List[Dict[str, Any]],
    visited_stack: Set[int],
    max_children_per_parent: Optional[int] = None,
) -> None:
    if parent_table_idx in visited_stack:
        return
    visited_stack.add(parent_table_idx)

    children = get_children_fks(schema, parent_table_idx)
    if not children:
        visited_stack.remove(parent_table_idx)
        return

    for fk in children:
        child_idx = fk.src_table
        child_table_name = schema.table_names[child_idx]
        child_fk_col = fk.src_col_name
        parent_pk_col = fk.dst_col_name

        if not sqlite_table_exists(conn, child_table_name):
            for pr in parent_rows:
                pr[child_table_name] = []
            continue

        child_rows_all = sqlite_fetch_all(conn, child_table_name)

        index = defaultdict(list)
        for r in child_rows_all:
            index[r.get(child_fk_col)].append(r)

        for pr in parent_rows:
            key = pr.get(parent_pk_col)
            kids = index.get(key, [])
            if max_children_per_parent is not None:
                kids = kids[:max_children_per_parent]
            pr[child_table_name] = kids
            embed_children(schema, conn, child_idx, pr[child_table_name], visited_stack, max_children_per_parent)

    visited_stack.remove(parent_table_idx)


def export_nested_docs_all_dbs(
    schemas: Dict[str, Schema],
    sqlite_root: str,
    out_nested_jsonl: str,
    max_children: Optional[int] = None,
    db_id_filter: Optional[Set[str]] = None,
) -> None:
    os.makedirs(os.path.dirname(out_nested_jsonl) or ".", exist_ok=True)
    with open(out_nested_jsonl, "w", encoding="utf-8") as out_f:
        for db_id, schema in schemas.items():
            if db_id_filter is not None and db_id not in db_id_filter:
                continue
            sqlite_path = os.path.join(sqlite_root, db_id, f"{db_id}.sqlite")
            if not os.path.exists(sqlite_path):
                continue

            conn = sqlite3.connect(sqlite_path)
            conn.row_factory = sqlite3.Row

            graph = build_table_graph(schema)
            comps = connected_components(graph)

            for comp in comps:
                if detect_fk_cycles(schema, comp):
                    continue
                root_idx = choose_main_table(schema, comp)
                root_table = schema.table_names[root_idx]
                if not sqlite_table_exists(conn, root_table):
                    continue

                root_rows = sqlite_fetch_all(conn, root_table)
                embed_children(schema, conn, root_idx, root_rows, visited_stack=set(), max_children_per_parent=max_children)

                for doc in root_rows:
                    out_f.write(json.dumps({
                        "db_id": db_id,
                        "root_table": root_table,
                        "_root_collection": f"{db_id}__{root_table}",
                        "doc": doc
                    }, ensure_ascii=False) + "\n")

            conn.close()


# ============================================================
# main
# ============================================================

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-json", required=True)
    ap.add_argument("--train-gold-sql", required=True)
    ap.add_argument("--train-tables", required=True)
    ap.add_argument("--out-jsonl", required=True, help="Output gold-only JSONL with SQL + placeholder mongo")

    # optional: algorithm-1 nested export
    ap.add_argument("--sqlite-root", default=None, help="If provided, enable Algorithm-1 nested export")
    ap.add_argument("--out-nested-jsonl", default=None, help="Where to write nested docs JSONL (requires --sqlite-root)")
    ap.add_argument("--max-children", type=int, default=None, help="Cap child rows per parent in nesting")

    # optional: process only db_ids present in train.json (usually best)
    ap.add_argument("--only-train-dbs", action="store_true", help="Limit nested export to db_ids appearing in train.json")

    args = ap.parse_args()

    train_items = load_train_json(args.train_json)
    gold_sql = load_gold_sql_lines(args.train_gold_sql)

    # 1) Always generate gold-only JSONL aligned with train.json
    build_gold_jsonl(train_items, gold_sql, args.out_jsonl)
    print(f"[OK] wrote gold-only JSONL: {args.out_jsonl}")
    print(f"     items: {len(train_items)}")

    # 2) Optional: Algorithm-1 nested export
    if args.sqlite_root and args.out_nested_jsonl:
        schemas = load_schemas(args.train_tables)

        db_id_filter = None
        if args.only_train_dbs:
            db_id_filter = {it["db_id"] for it in train_items if "db_id" in it}

        export_nested_docs_all_dbs(
            schemas=schemas,
            sqlite_root=args.sqlite_root,
            out_nested_jsonl=args.out_nested_jsonl,
            max_children=args.max_children,
            db_id_filter=db_id_filter
        )
        print(f"[OK] wrote nested docs JSONL (Algorithm-1): {args.out_nested_jsonl}")

    elif args.sqlite_root or args.out_nested_jsonl:
        print("[WARN] To export nested docs, provide BOTH --sqlite-root and --out-nested-jsonl")


if __name__ == "__main__":
    main()

