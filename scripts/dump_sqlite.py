
import sqlite3
import os
from pathlib import Path

def dump_sqlite_to_sql(db_path, output_dir):
    """Dumps a SQLite DB to a series of INSERT statements."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    db_name = Path(db_path).stem
    out_file = output_dir / f"{db_name}_dump.sql"
    
    print(f"Dumping {db_name} to {out_file}...")
    
    with open(out_file, 'w', encoding='utf-8') as f:
        for table_name in tables:
            t_name = table_name[0]
            if t_name == "sqlite_sequence": continue
            
            # Get data
            cursor.execute(f"SELECT * FROM {t_name}")
            rows = cursor.fetchall()
            
            if not rows: continue
            
            # Simple INSERT generation
            f.write(f"-- Table: {t_name}\n")
            for row in rows:
                # Format values safely
                formatted_values = []
                for item in row:
                    if item is None:
                        formatted_values.append("NULL")
                    elif isinstance(item, str):
                        clean_str = item.replace("'", "''")
                        formatted_values.append(f"'{clean_str}'")
                    else:
                        formatted_values.append(str(item))
                
                vals = ", ".join(formatted_values)
                sql = f"INSERT INTO {t_name} VALUES ({vals});\n"
                f.write(sql)
            f.write("\n")
            
    conn.close()
    return out_file

def run_dump_all():
    base_dir = Path("data/bird/minidev/MINIDEV/dev_databases")
    dump_dir = Path("data/bird/sql_dumps")
    dump_dir.mkdir(parents=True, exist_ok=True)
    
    if not base_dir.exists():
        print("BirdBench databases not found.")
        return

    count = 0
    for db_dir in base_dir.iterdir():
        if db_dir.is_dir():
            db_path = db_dir / f"{db_dir.name}.sqlite"
            if db_path.exists():
                dump_sqlite_to_sql(db_path, dump_dir)
                count += 1
                
    print(f"Dumped {count} databases to SQL scripts.")

if __name__ == "__main__":
    run_dump_all()
