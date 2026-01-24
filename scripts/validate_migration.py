import os
import sys
import pandas as pd
from sqlalchemy import create_engine
from pymongo import MongoClient

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from services.migration.schema_discovery import SchemaDiscovery

MONGO_URI = "mongodb://localhost:27017"
BASE_DIR = os.path.join("data", "minidev", "MINIDEV", "dev_databases")

def validate_all():
    mongo_client = MongoClient(MONGO_URI)
    
    if not os.path.exists(BASE_DIR):
        print(f"Error: Base directory not found at {BASE_DIR}")
        return

    # Open file for writing
    with open("validation_report.txt", "w") as f:
        header = f"{'DATABASE':<25} | {'TABLE':<20} | {'SQLITE':<10} | {'MONGO':<10} | {'STATUS':<10}"
        separator = "-" * 85
        
        print(header)
        print(separator)
        f.write(header + "\n")
        f.write(separator + "\n")

        all_match = True
        mismatches = []

        for db_id in os.listdir(BASE_DIR):
            db_path = os.path.join(BASE_DIR, db_id)
            if os.path.isdir(db_path):
                sqlite_file = os.path.join(db_path, f"{db_id}.sqlite")
                
                if os.path.exists(sqlite_file):
                    # Connect to SQLite
                    engine = create_engine(f"sqlite:///{sqlite_file}")
                    inspector = SchemaDiscovery(sqlite_file)
                    schema = inspector.get_schema()
                    
                    # Connect to Mongo DB
                    mongo_db = mongo_client[db_id]
                    
                    for table_name in schema.keys():
                        # Count SQLite
                        try:
                            sqlite_count = pd.read_sql_query(f"SELECT COUNT(*) FROM {table_name}", engine).iloc[0, 0]
                        except Exception as e:
                            line = f"{db_id:<25} | {table_name:<20} | ERROR      | -          | {str(e)}"
                            print(line)
                            f.write(line + "\n")
                            continue

                        # Count Mongo
                        mongo_count = mongo_db[table_name].count_documents({})
                        
                        status = "MATCH" if sqlite_count == mongo_count else "MISMATCH"
                        if status == "MISMATCH":
                            all_match = False
                            mismatches.append((db_id, table_name, sqlite_count, mongo_count))
                        
                        line = f"{db_id:<25} | {table_name:<20} | {sqlite_count:<10} | {mongo_count:<10} | {status:<10}"
                        print(line)
                        f.write(line + "\n")

        print(separator)
        f.write(separator + "\n")
        
        if all_match:
            msg = "\nSUCCESS: All tables and row counts match exactly!"
            print(msg)
            f.write(msg + "\n")
        else:
            msg = f"\nWARNING: {len(mismatches)} table(s) have discrepancies:"
            print(msg)
            f.write(msg + "\n")
            for db, table, sq_count, mg_count in mismatches:
                line = f"  - {db}.{table}: SQLite={sq_count}, Mongo={mg_count}"
                print(line)
                f.write(line + "\n")

if __name__ == "__main__":
    validate_all()
