import json
import os
import random

def generate_ddl_data():
    """Generates synthetic DDL/DML data for training text-to-SQL models."""
    
    data = []
    
    # helper to add example
    def add(q, sql):
        data.append({
            "question": q,
            "query": sql,
            "db_id": "ddl_common", # Generic ID
            "complexity": "simple"
        })

    # --- CREATE TABLE ---
    templates = [
        ("Create a table named {table} with columns {c1} and {c2}", "CREATE TABLE {table} ({c1} VARCHAR, {c2} INTEGER);"),
        ("New table {table} containing {c1}, {c2}, {c3}", "CREATE TABLE {table} ({c1} INTEGER, {c2} VARCHAR, {c3} VARCHAR);"),
        ("Initialize table {table} with fields {c1} (int) and {c2} (text)", "CREATE TABLE {table} ({c1} INTEGER, {c2} TEXT);"),
        ("Create table {table}", "CREATE TABLE {table} (id INTEGER PRIMARY KEY);")
    ]
    
    tables = ["employees", "users", "products", "orders", "customers", "students", "courses", "inventory"]
    cols = ["name", "age", "email", "status", "date", "price", "description", "category", "address"]
    
    for _ in range(50):
        t = random.choice(templates)
        table = random.choice(tables)
        c1 = random.choice(cols)
        c2 = random.choice([c for c in cols if c != c1])
        c3 = random.choice([c for c in cols if c not in [c1, c2]])
        
        q = t[0].format(table=table, c1=c1, c2=c2, c3=c3)
        sql = t[1].format(table=table, c1=c1, c2=c2, c3=c3)
        add(q, sql)
        
    # --- INSERT ---
    for _ in range(50):
        table = random.choice(tables)
        col = random.choice(cols)
        val = "'John'" if col == 'name' else "25" if col == 'age' else "'active'"
        
        add(f"Insert {val} into {table}", f"INSERT INTO {table} VALUES ({val});")
        add(f"Add a new record to {table} with {col} as {val}", f"INSERT INTO {table} ({col}) VALUES ({val});")
        
    # --- UPDATE ---
    for _ in range(30):
        table = random.choice(tables)
        col = random.choice(cols)
        val = "'Completed'"
        cond_col = random.choice(cols)
        cond_val = "5"
        
        add(f"Update {table} set {col} to {val}", f"UPDATE {table} SET {col} = {val};")
        add(f"Change {col} to {val} in {table} where {cond_col} is {cond_val}", f"UPDATE {table} SET {col} = {val} WHERE {cond_col} = {cond_val};")
        
    # --- DROP ---
    for table in tables:
        add(f"Drop table {table}", f"DROP TABLE {table};")
        add(f"Delete table {table}", f"DROP TABLE {table};")
        
    # --- DELETE ---
    for _ in range(30):
        table = random.choice(tables)
        col = random.choice(cols)
        val = "10"
        add(f"Delete from {table}", f"DELETE FROM {table};")
        add(f"Remove rows from {table} where {col} is {val}", f"DELETE FROM {table} WHERE {col} = {val};")

    # Output
    output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'ddl_tasks.json')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"Generated {len(data)} DDL/DML examples at {output_path}")

if __name__ == "__main__":
    generate_ddl_data()
