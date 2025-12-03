# src/evaluation/metrics.py

import re


def normalize_sql(sql: str) -> str:
    """
    Normalize SQL for comparison:
    - Lowercase
    - Collapse whitespace
    - Strip leading/trailing spaces
    """
    if sql is None:
        return ""
    sql = sql.lower()
    sql = re.sub(r"\s+", " ", sql)
    sql = sql.strip()
    return sql


def exact_match(pred_sql: str, gold_sql: str) -> bool:
    """
    Exact Match Accuracy:
    True if normalized predicted SQL equals normalized gold SQL.
    """
    return normalize_sql(pred_sql) == normalize_sql(gold_sql)


def valid_efficiency_score(pred_sql: str, gold_sql: str) -> float:
    """
    Valid Efficiency Score (simplified):

    - If prediction does NOT exactly match -> 0
    - If matches -> 1 / (#tokens in prediction)

    This rewards shorter correct SQL queries.
    """
    if not exact_match(pred_sql, gold_sql):
        return 0.0

    tokens = pred_sql.split()
    if not tokens:
        return 0.0

    return 1.0 / len(tokens)
