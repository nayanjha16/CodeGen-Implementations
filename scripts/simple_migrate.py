import os
import sys
import pandas as pd
from sqlalchemy import create_engine
from pymongo import MongoClient

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

MONGO_URI = "mongodb://localhost:27017"
BASE_DIR = os.path.join("data", "minidev", "MINIDEV", "dev_databases")

# Failed tables from validation
failed_tables = [
    ("california_schools", "schools"),
    ("codebase_community", "posts"),
    ("formula_1", "drivers"),
    ("thrombosis_prediction", "Examination"),
    ("thrombosis_prediction", "Patient"),
]

def simple_migrate():
    client = MongoClient(MONGO_URI)
    
    for db_id, table_name in failed_tables:
        sqlite_file = os.path.join(BASE_DIR, db_id, f"{db_id}.sqlite")
        
        if not os.path.exists(sqlite_file):
            print(f"ERROR: SQLite file not found: {sqlite_file}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Migrating: {db_id}.{table_name}")
        print(f"{'='*60}")
        
        try:
            engine = create_engine(f"sqlite:///{sqlite_file}")
            mongo_db = client[db_id]
            
            # Clear existing
            print(f"Clearing existing data...")
            mongo_db[table_name].delete_many({})
            
            # Read in chunks WITHOUT any type conversion
            chunk_size = 50000
            total_records = 0
            
            print(f"Reading from SQLite (NO type inference)...")
            
            # Use SQL query instead of table read to control types better
            query = f"SELECT * FROM {table_name}"
            
            for chunk_df in pd.read_sql_query(query, engine, chunksize=chunk_size):
                if chunk_df.empty:
                    continue
                
                # Convert NA/NaN to None, but do NOT touch datatypes
                chunk_df = chunk_df.where(pd.notnull(chunk_df), None)
                
                # Convert to dict
                records = chunk_df.to_dict(orient='records')
                
                if records:
                    mongo_db[table_name].insert_many(records)
                    total_records += len(records)
                    print(f"  Inserted {len(records)} records (Total: {total_records})")
            
            print(f"SUCCESS: Migrated {total_records} records to {db_id}.{table_name}")
            
        except Exception as e:
            print(f"FAILED: {db_id}.{table_name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    simple_migrate()
