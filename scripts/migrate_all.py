import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.migration.migrate import MigrationService

MONGO_URI = "mongodb://localhost:27018"
BASE_DIR = os.path.join("data", "minidev", "MINIDEV", "dev_databases")

def migrate_all():
    if not os.path.exists(BASE_DIR):
        print(f"Error: Base directory not found at {BASE_DIR}")
        return

    # List all subdirectories (database IDs)
    for db_id in os.listdir(BASE_DIR):
        db_path = os.path.join(BASE_DIR, db_id)
        if os.path.isdir(db_path):
            sqlite_file = os.path.join(db_path, f"{db_id}.sqlite")
            
            if os.path.exists(sqlite_file):
                print(f"\n==========================================")
                print(f"Migrating database: {db_id}")
                print(f"Source: {sqlite_file}")
                print(f"Target: MongoDB (db: {db_id})")
                print(f"==========================================")
                
                try:
                    service = MigrationService(sqlite_file, MONGO_URI, db_id)
                    service.migrate()
                    print(f"Successfully migrated {db_id}")
                except Exception as e:
                    print(f"FAILED to migrate {db_id}: {e}")
            else:
                print(f"Skipping {db_id}: No sqlite file found at {sqlite_file}")

if __name__ == "__main__":
    migrate_all()
