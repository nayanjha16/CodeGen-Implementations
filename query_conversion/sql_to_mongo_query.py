"""Convert a subset of SQL queries into MongoDB aggregation pipelines.

Supports SELECT queries with WHERE, GROUP BY, HAVING, ORDER BY and simple
aggregates (COUNT, SUM, AVG, MIN, MAX). Produces pipelines using stages:
  - $match
  - $group
  - $project
  - $unwind (when UNNEST/UNWIND present)
  - $sort

The converter is intentionally conservative and handles simple SQL forms.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


def split_comma_outside_parens(s: str) -> List[str]:
    parts = []
    cur = []
    depth = 0
    for ch in s:
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        if ch == ',' and depth == 0:
            parts.append(''.join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    if cur:
        parts.append(''.join(cur).strip())
    return parts


def parse_select_list(select_clause: str) -> List[Tuple[str, Optional[str]]]:
    """Return list of (expression, alias) pairs for items in SELECT clause."""
    items = split_comma_outside_parens(select_clause)
    res = []
    for it in items:
        m = re.match(r"(.*)\s+as\s+(\w+)$", it, flags=re.I)
        if m:
            expr = m.group(1).strip()
            alias = m.group(2)
        else:
            # alias might be trailing token
            parts = it.rsplit(None, 1)
            if len(parts) == 2 and re.match(r"^[a-zA-Z_]\w*$", parts[1]):
                expr, alias = parts[0], parts[1]
            else:
                expr, alias = it, None
        res.append((expr.strip(), alias))
    return res


def parse_order_by(order_clause: str) -> List[Tuple[str, int]]:
    items = split_comma_outside_parens(order_clause)
    res = []
    for it in items:
        m = re.match(r"(\S+)(?:\s+(asc|desc))?", it.strip(), flags=re.I)
        if m:
            col = m.group(1)
            dirn = m.group(2)
            res.append((col, -1 if dirn and dirn.lower() == 'desc' else 1))
    return res


def parse_where_clause(where: str) -> Dict[str, Any]:
    """Very small parser for simple WHERE expressions like `a = 1` or `a IN (1,2)`.

    Supports AND of simple comparisons.
    """
    # Split by AND (very naive)
    conds = [c.strip() for c in re.split(r"\band\b", where, flags=re.I) if c.strip()]
    query: Dict[str, Any] = {}
    for c in conds:
        m_in = re.match(r"^(\w+)\s+in\s*\((.*)\)$", c, flags=re.I)
        if m_in:
            col = m_in.group(1)
            vals = [v.strip().strip("'\"") for v in m_in.group(2).split(',')]
            # convert numeric
            vals2 = [int(v) if re.match(r"^-?\d+$", v) else float(v) if re.match(r"^-?\d+\.\d+$", v) else v for v in vals]
            query[col] = {"$in": vals2}
            continue

        m_cmp = re.match(r"^(\w+)\s*(=|<>|!=|>=|<=|>|<)\s*(.+)$", c)
        if m_cmp:
            col, op, val = m_cmp.group(1), m_cmp.group(2), m_cmp.group(3).strip()
            # strip quotes
            if val.startswith(("'", '"')) and val.endswith(("'", '"')):
                val_parsed = val[1:-1]
            elif re.match(r"^-?\d+$", val):
                val_parsed = int(val)
            elif re.match(r"^-?\d+\.\d+$", val):
                val_parsed = float(val)
            else:
                val_parsed = val

            if op in ("=",):
                query[col] = val_parsed
            elif op in ("!=", "<>"):
                query[col] = {"$ne": val_parsed}
            elif op == ">":
                query[col] = {"$gt": val_parsed}
            elif op == "<":
                query[col] = {"$lt": val_parsed}
            elif op == ">=":
                query[col] = {"$gte": val_parsed}
            elif op == "<=":
                query[col] = {"$lte": val_parsed}
            continue

    return query


def sql_to_mongo_pipeline(sql: str) -> Optional[Dict[str, Any]]:
    """Convert a single SQL query into a MongoDB pipeline dict:
    {"collection": <table>, "pipeline": [stages], "order_by": [...]}
    Returns None if query cannot be handled.
    """
    s = sql.strip().rstrip(';')
    # Extract clauses
    m_select = re.search(r"select\s+(.*?)\s+from\s+", s, flags=re.I | re.S)
    if not m_select:
        return None
    select_clause = m_select.group(1).strip()

    m_from = re.search(r"from\s+([a-zA-Z_][\.\w]*)", s, flags=re.I)
    if not m_from:
        return None
    collection = m_from.group(1).split('.')[-1]

    m_where = re.search(r"where\s+(.*?)\s*(group by|having|order by|$)", s, flags=re.I | re.S)
    where_clause = m_where.group(1).strip() if m_where else None

    m_group = re.search(r"group\s+by\s+(.*?)\s*(having|order by|$)", s, flags=re.I | re.S)
    group_clause = m_group.group(1).strip() if m_group else None

    m_having = re.search(r"having\s+(.*?)\s*(order by|$)", s, flags=re.I | re.S)
    having_clause = m_having.group(1).strip() if m_having else None

    m_order = re.search(r"order\s+by\s+(.*)$", s, flags=re.I | re.S)
    order_clause = m_order.group(1).strip() if m_order else None

    select_items = parse_select_list(select_clause)

    pipeline: List[Dict[str, Any]] = []

    # WHERE -> $match
    if where_clause:
        match_q = parse_where_clause(where_clause)
        if match_q:
            pipeline.append({"$match": match_q})

    # Determine aggregates in select
    aggregates = {}
    projections = []  # list of (out_name, expr)
    for expr, alias in select_items:
        agg_m = re.match(r"(count|sum|avg|min|max)\s*\(\s*(\*|[\w\.]+)\s*\)\s*$", expr, flags=re.I)
        if agg_m:
            func = agg_m.group(1).lower()
            inner = agg_m.group(2)
            out_name = alias or f"{func}"
            if func == 'count':
                aggregates[out_name] = {"$sum": 1}
            elif func == 'sum':
                aggregates[out_name] = {"$sum": f"${inner}"}
            elif func == 'avg':
                aggregates[out_name] = {"$avg": f"${inner}"}
            elif func == 'min':
                aggregates[out_name] = {"$min": f"${inner}"}
            elif func == 'max':
                aggregates[out_name] = {"$max": f"${inner}"}
            projections.append((out_name, None))
        else:
            # non-aggregate expression assumed to be group key or projection
            col = expr
            out_name = alias or col
            projections.append((out_name, col))

    group_by_fields = []
    if group_clause:
        group_by_fields = [g.strip() for g in group_clause.split(',')]

    if aggregates or group_by_fields:
        # build $group
        group_stage: Dict[str, Any] = {"_id": {}}
        if group_by_fields:
            for fld in group_by_fields:
                # map to $field
                key = fld
                group_stage["_id"][fld] = f"${key}"
        else:
            group_stage["_id"] = None

        for out, acc in aggregates.items():
            group_stage[out] = acc

        pipeline.append({"$group": group_stage})

        # HAVING -> post-group $match
        if having_clause:
            # very naive: replace SQL operators to Mongo expressions for simple conditions like 'cnt > 1'
            # assume form: <agg_name> <op> <value>
            m = re.match(r"(\w+)\s*(=|<>|!=|>=|<=|>|<)\s*(.+)$", having_clause)
            if m:
                fld, op, val = m.group(1), m.group(2), m.group(3).strip()
                # convert val
                if val.startswith(("'", '"')) and val.endswith(("'", '"')):
                    valp = val[1:-1]
                elif re.match(r"^-?\d+$", val):
                    valp = int(val)
                elif re.match(r"^-?\d+\.\d+$", val):
                    valp = float(val)
                else:
                    valp = val

                op_map = {"=": "$eq", "!=": "$ne", "<>": "$ne", ">": "$gt", "<": "$lt", ">=": "$gte", "<=": "$lte"}
                mongo_op = op_map.get(op)
                if mongo_op:
                    pipeline.append({"$match": {fld: {mongo_op: valp}}})

        # project: move _id fields back to top-level
        proj: Dict[str, Any] = {}
        for out, col in projections:
            if col is None:
                proj[out] = 1
            else:
                # column likely in _id
                proj[out] = f"$_id.{col}" if group_by_fields else f"${col}"

        proj["_id"] = 0
        pipeline.append({"$project": proj})

    else:
        # no group: project selected fields if not *
        # detect select *
        if select_clause.strip() != '*':
            proj = {}
            for expr, alias in select_items:
                proj_name = alias or expr
                proj[proj_name] = f"${expr}" if '.' not in expr else f"${expr.split('.')[-1]}"
            pipeline.append({"$project": proj})

    # ORDER BY -> $sort
    order_by = []
    if order_clause:
        ob = parse_order_by(order_clause)
        sort_stage = {k: v for k, v in ob}
        pipeline.append({"$sort": sort_stage})
        order_by = ob

    return {"collection": collection, "pipeline": pipeline, "order_by": order_by}


if __name__ == "__main__":
    # Quick manual test
    q = "SELECT a, COUNT(*) as cnt FROM t WHERE x = 1 GROUP BY a HAVING cnt > 1 ORDER BY cnt DESC;"
    print(sql_to_mongo_pipeline(q))
