import os
import sys
import pandas as pd
from pymongo import MongoClient

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from services.migration.migrate import MigrationService

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

def retry_failed():
    client = MongoClient(MONGO_URI)
    
    for db_id, table_name in failed_tables:
        sqlite_file = os.path.join(BASE_DIR, db_id, f"{db_id}.sqlite")
        
        if not os.path.exists(sqlite_file):
            print(f"ERROR: SQLite file not found: {sqlite_file}")
            continue
        
        print(f"\n{'='*60}")
        print(f"Retrying: {db_id}.{table_name}")
        print(f"{'='*60}")
        
        try:
            service = MigrationService(sqlite_file, MONGO_URI, db_id)
            mongo_db = client[db_id]
            
            # Get schema for this table
            schema = service.discovery.get_schema()
            if table_name not in schema:
                print(f"ERROR: Table {table_name} not found in schema")
                continue
            
            columns = schema[table_name]
            
            # Clear existing
            print(f"Clearing existing data...")
            mongo_db[table_name].delete_many({})
            
            # Read and migrate in chunks
            chunk_size = 50000
            total_records = 0
            
            print(f"Reading from SQLite...")
            
            try:
                for i, chunk_df in enumerate(pd.read_sql_table(table_name, service.engine, chunksize=chunk_size)):
                    print(f"  Processing chunk {i+1}...")
                    
                    if chunk_df.empty:
                        continue
                    
                    try:
                        # Type Inference
                        print(f"    Type inference...")
                        chunk_df = service.infer_and_convert_types(chunk_df, columns)
                    except Exception as e:
                        print(f"    ERROR in type inference: {e}")
                        raise
                    
                    try:
                        # Convert to dict
                        print(f"    Converting to dict...")
                        records = chunk_df.to_dict(orient='records')
                    except Exception as e:
                        print(f"    ERROR in to_dict: {e}")
                        print(f"    Datetime columns: {[c for c in chunk_df.columns if pd.api.types.is_datetime64_any_dtype(chunk_df[c])]}")
                        raise
                    
                    if records:
                        print(f"    Inserting {len(records)} records...")
                        mongo_db[table_name].insert_many(records)
                        total_records += len(records)
                        print(f"  Inserted {len(records)} records (Total: {total_records})")
            except Exception as e:
                print(f"  ERROR during processing: {e}")
                raise
            
            print(f"SUCCESS: Migrated {total_records} records to {db_id}.{table_name}")
            
        except Exception as e:
            print(f"FAILED: {db_id}.{table_name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    retry_failed()
