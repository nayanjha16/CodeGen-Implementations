
import os
import sqlite3
import json
from pathlib import Path

def verify_bird_databases():
    # Path to the downloaded databases
    base_dir = Path("data/bird/minidev/MINIDEV/dev_databases")
    
    if not base_dir.exists():
        print(f"Database directory not found: {base_dir}")
        return

    print(f"Checking databases in {base_dir}...")
    
    valid_count = 0
    error_count = 0
    
    for db_folder in base_dir.iterdir():
        if db_folder.is_dir():
            db_name = db_folder.name
            db_path = db_folder / f"{db_name}.sqlite"
            
            if db_path.exists():
                try:
                    conn = sqlite3.connect(str(db_path))
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = cursor.fetchall()
                    conn.close()
                    
                    if len(tables) > 0:
                        # print(f"✓ {db_name}: {len(tables)} tables")
                        valid_count += 1
                    else:
                        print(f"⚠ {db_name}: Connected but no tables found.")
                        error_count += 1
                        
                except Exception as e:
                    print(f"✗ {db_name}: Error connecting - {e}")
                    error_count += 1
            else:
                print(f"✗ {db_name}: SQLite file missing at {db_path}")
                error_count += 1
                
    print(f"\nVerification Complete.")
    print(f"Valid Databases: {valid_count}")
    print(f"Errors/Missing: {error_count}")

if __name__ == "__main__":
    verify_bird_databases()
