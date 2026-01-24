import os
import sys

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from services.migration.migrate import MigrationService

MONGO_URI = "mongodb://localhost:27017"

def test_financial_migration():
    # Path to financial.sqlite
    # Adjusted to match the structure we found
    base_dir = "data/minidev/MINIDEV/dev_databases"
    db_id = "financial"
    sqlite_path = os.path.join(base_dir, db_id, f"{db_id}.sqlite")
    
    if not os.path.exists(sqlite_path):
        print(f"Error: Database file not found at {sqlite_path}")
        return

    print(f"Testing migration for {db_id}...")
    service = MigrationService(sqlite_path, MONGO_URI, db_id)
    service.migrate()
    print("Test Complete.")

if __name__ == "__main__":
    test_financial_migration()
