import sqlite3


def extract_sqlite_schema(sqlite_db_path: str) -> str:
    """
    Extract SQLite schema as plain text for LLM prompting.
    """
    conn = sqlite3.connect(sqlite_db_path)
    cursor = conn.cursor()

    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table';"
    )
    tables = [row[0] for row in cursor.fetchall()]

    schema_lines = []

    for table in tables:
        cursor.execute(f"PRAGMA table_info({table});")
        columns = cursor.fetchall()
        col_defs = ", ".join([col[1] for col in columns])
        schema_lines.append(f"{table}({col_defs})")

    conn.close()
    return "\n".join(schema_lines)
