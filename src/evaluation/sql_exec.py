# src/evaluation/sql_exec.py

import os
import sqlite3
from typing import Tuple, Any


def execute_query(db_path: str, sql_query: str) -> Tuple[Any, bool]:
    """
    Execute a single SQL query against a SQLite database.

    Returns:
      (rows, True)  if execution succeeds
      (None, False) if it fails
    """
    if not os.path.exists(db_path):
        return None, False

    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql_query)
        rows = cur.fetchall()
        conn.close()
        return rows, True
    except Exception:
        return None, False


def execution_accuracy(
    pred_sql: str,
    gold_sql: str,
    db_id: str,
    spider_db_root: str = "data/spider/database",
) -> bool:
    """
    Execution Accuracy (EX):

    - Build the DB file path for the given db_id.
    - Execute both predicted and gold SQL.
    - Return True if results are identical and both queries succeed.
    """
    db_file = os.path.join(spider_db_root, db_id, f"{db_id}.sqlite")

    pred_out, pred_ok = execute_query(db_file, pred_sql)
    gold_out, gold_ok = execute_query(db_file, gold_sql)

    if not pred_ok or not gold_ok:
        return False

    return pred_out == gold_out


def test_suite_accuracy(
    pred_sql: str,
    gold_sql: str,
    db_id: str,
    spider_db_root: str = "data/spider/database",
) -> bool:
    """
    Test-Suite Accuracy (TSA) – APPROXIMATION:

    True TSA requires multiple test databases / edge-case checks per db_id.
    For now we approximate TSA as execution accuracy on the main DB.
    """
    return execution_accuracy(pred_sql, gold_sql, db_id, spider_db_root)
