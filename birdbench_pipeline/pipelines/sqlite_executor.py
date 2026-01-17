import sqlite3
from typing import List, Dict, Any


def execute_sqlite_query(db_path: str, sql: str) -> List[Dict[str, Any]]:
    """
    Execute a SQL query on a SQLite database and return results
    as a list of dictionaries: [{column: value, ...}, ...]
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enables dict-style rows

    try:
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()

        results = [dict(row) for row in rows]
        return results

    finally:
        conn.close()
