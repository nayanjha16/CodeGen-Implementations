import sqlite3
import json
from pathlib import Path

def setup_sql_db(file_path):
    # 1. Initialize the mock client and database
    conn = sqlite3.connect(file_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    return cursor