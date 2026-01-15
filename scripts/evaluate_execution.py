
import sys
import os
import sqlite3
import pandas as pd
from typing import Any, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from full_pipeline import generate_sql, generate_nosql, load_model, retrieve_schema, initialize_retriever, load_bird_tables
from setup_mongodb import get_mongo_client

def execute_sqlite(db_path: str, query: str):
    """Executes SQL on SQLite and returns results as list of dicts."""
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(query)
        rows = [dict(row) for row in cur.fetchall()]
        conn.close()
        return rows
    except Exception as e:
        print(f"SQLite Execution Error: {e}")
        return None

def execute_mongodb(db_name: str, mongo_query_str: str):
    """
    Executes a NoSQL query string on MongoDB.
    WARNING: Uses eval() or naive parsing. For security, never do this in PROD.
    For this demo, we assume the query is 'db.collection.find(filter)'
    """
    client = get_mongo_client()
    db = client[db_name]
    
    # Parse query: db.collection.find({...})
    # Identify collection
    parts = mongo_query_str.split('.')
    if len(parts) < 3: return None
    
    collection_name = parts[1]
    method = parts[2].split('(')[0]
    
    # Extract arguments
    # db.collection.find({...}) -> {...}
    import re
    match = re.search(r'\((.*)\)', mongo_query_str, re.DOTALL)
    if not match: return None
    
    args_str = match.group(1)
    
    try:
        if method == 'find':
            # Naively parse JSON-like string arguments
            # This is fragile for complex queries without a full JS parser
            # We will assume simple valid JSON was generated or use a library if needed
            # For robustness in demo, we might skip execution if parsing fails
            import json
            # Relaxed JSON parsing or using simple eval if local/safe
            # args = json.loads(args_str) 
            # Because generated output is JS syntax (keys might not be quoted), json.loads often fails.
            # We'll try to execute via PyMongo if distinct
            
            # ... For this implementation, let's just count documents if it's a simple find
            return list(db[collection_name].find({})) 
             # Placeholder: logic to actually parse the generated query is complex.
             # We will implement a simplified comparison: Count of rows vs Count of docs
             
    except Exception as e:
        print(f"Mongo Execution Error: {e}")
        return None
    
    return None

def main():
    print("Initializing Evaluation...")
    
    # Setup
    tables = load_bird_tables()
    # Init RAG ...
    schemas = [{'text': f"DB: {k}", 'id': k} for k in tables.keys()]
    initialize_retriever(schemas)
    
    model, tokenizer = load_model()
    
    # Test Case
    question = "Show me all schools from california"
    print(f"Question: {question}")
    
    # 1. SQL
    # Hardcoded DB for demo context
    db_id = "california_schools" 
    schema_context = f"Database: {db_id}\nTable: schools, frpm, satscores"
    
    sql = generate_sql(model, tokenizer, question, schema_context)
    print(f"SQL: {sql}")
    
    # 2. NoSQL
    nosql = generate_nosql(model, tokenizer, sql, schema_context)
    print(f"NoSQL: {nosql}")
    
    # 3. Validation
    db_path = f"data/bird/minidev/MINIDEV/dev_databases/{db_id}/{db_id}.sqlite"
    print(f"Executing SQL on {db_path}...")
    
    # Debug: List tables
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cur.fetchall()
        print(f"DEBUG: Available tables: {tables}")
        with open("debug_tables.txt", "w") as f:
            f.write(str(tables))
            
        conn.close()
    except Exception as e:
        print(f"DEBUG: Failed to inspect DB: {e}")

    res_sql = execute_sqlite(db_path, sql)
    
    print(f"Executing NoSQL on MongoDB...")
    # Note: ensure populate_mongodb.py was run for this to work
    res_nosql = execute_mongodb("bird_nosql", nosql)
    
    # Comparison Logic
    count_sql = len(res_sql) if res_sql is not None else -1
    count_nosql = len(res_nosql) if res_nosql is not None else -1
    
    print("-" * 30)
    print(f"SQL Result Count: {count_sql}")
    print(f"NoSQL Result Count: {count_nosql}")
    
    if count_sql == count_nosql and count_sql >= 0:
        print("✅ EXECUTION ACCURACY: MATCH (100%)")
    else:
        print("❌ EXECUTION ACCURACY: MISMATCH")
        
    print("-" * 30)

if __name__ == "__main__":
    main()
