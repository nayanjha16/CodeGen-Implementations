import os
import sys
import pandas as pd
from sqlalchemy import create_engine

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from services.migration.schema_discovery import SchemaDiscovery

BASE_DIR = os.path.join("data", "minidev", "MINIDEV", "dev_databases")

# Failed tables from validation
failed_tables = [
    ("california_schools", "schools"),
    ("codebase_community", "posts"),
    ("formula_1", "drivers"),
    ("thrombosis_prediction", "Examination"),
    ("thrombosis_prediction", "Patient"),
]

def diagnose_tables():
    for db_id, table_name in failed_tables:
        sqlite_file = os.path.join(BASE_DIR, db_id, f"{db_id}.sqlite")
        
        if not os.path.exists(sqlite_file):
            print(f"ERROR: SQLite file not found: {sqlite_file}")
            continue
        
        print(f"\n{'='*70}")
        print(f"Diagnosing: {db_id}.{table_name}")
        print(f"{'='*70}")
        
        try:
            engine = create_engine(f"sqlite:///{sqlite_file}")
            
            # Read first chunk
            df = pd.read_sql_table(table_name, engine, chunksize=1000).__next__()
            
            print(f"Total columns: {len(df.columns)}")
            print(f"Row count (sample): {len(df)}")
            print("\nColumn datatypes:")
            
            datetime_cols = []
            for col in df.columns:
                dtype = df[col].dtype
                print(f"  {col:<30} -> {dtype}")
                
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    datetime_cols.append(col)
                    # Check for NaT
                    nat_count = df[col].isna().sum()
                    print(f"    ⚠️ DATETIME DETECTED! NaT count: {nat_count}/{len(df)}")
            
            if datetime_cols:
                print(f"\n❌ PROBLEM: {len(datetime_cols)} datetime column(s) found:")
                for col in datetime_cols:
                    print(f"  - {col}")
            else:
                print(f"\n✓ No datetime columns detected")
                
        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    diagnose_tables()
