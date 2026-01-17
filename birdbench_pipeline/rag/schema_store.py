import sqlite3

def load_schema(sqlite_db_path: str):
    """
    Load PK–FK relationships from SQLite DB.
    """
    conn = sqlite3.connect(sqlite_db_path)
    cursor = conn.cursor()

    schema_graph = {}

    cursor.execute("""
        SELECT m.name AS table_name, p.*
        FROM sqlite_master m
        JOIN pragma_foreign_key_list(m.name) p
        ON m.name != 'sqlite_sequence'
    """)

    for row in cursor.fetchall():
        table = row[0]
        ref_table = row[3]
        from_col = row[4]
        to_col = row[5]

        schema_graph.setdefault(table, []).append({
            "from_column": from_col,
            "to_table": ref_table,
            "to_column": to_col
        })

    conn.close()
    return schema_graph
