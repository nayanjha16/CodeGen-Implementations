"""
Script to migrate SQLite databases to JSON format for MongoDB simulation.

This script demonstrates the "SQL to NoSQL" data pipeline component. It reads
tables from SQLite databases (Spider/BirdBench) and converts them into
JSON documents that can be imported into MongoDB (or loaded into mongomock).
"""

import os
import sys
import json
import sqlite3
from tqdm import tqdm

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dataset_loader import load_spider_dataset, load_bird_dataset

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'mongo_dump')

def sqlite_to_json(db_path, db_id):
    """
    Convert a SQLite database to a dictionary of collections (tables).
    """
    if not os.path.exists(db_path):
        return None
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    mongo_db = {}
    
    for table in tables:
        # Skip internal tables
        if table.startswith('sqlite_'):
            continue
            
        cursor.execute(f"SELECT * FROM \"{table}\"")
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        
        documents = []
        for row in rows:
            doc = dict(zip(columns, row))
            documents.append(doc)
            
        mongo_db[table] = documents
        
    conn.close()
    return mongo_db

def main():
    print("Migrating SQLite databases to JSON (MongoDB format)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load combined DB list from both datasets
    # We use small limits just to get the db_paths, but for full migration 
    # we would loop through all unique DBs.
    spider_examples = load_spider_dataset(limit=10) # Just grabbing some to safeguard
    bird_examples = load_bird_dataset(limit=10)
    
    # Collect unique DBs
    unique_dbs = {}
    
    for ex in spider_examples:
        if ex.db_path and ex.db_id not in unique_dbs:
            unique_dbs[ex.db_id] = ex.db_path
            
    for ex in bird_examples:
        if ex.db_path and ex.db_id not in unique_dbs:
            unique_dbs[ex.db_id] = ex.db_path
            
    print(f"Found {len(unique_dbs)} unique databases to migrate in sample.")
    
    count = 0
    for db_id, db_path in tqdm(unique_dbs.items(), desc="Migrating"):
        try:
            mongo_data = sqlite_to_json(db_path, db_id)
            if mongo_data:
                output_file = os.path.join(OUTPUT_DIR, f"{db_id}.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(mongo_data, f, default=str) # default=str to handle dates/bytes
                count += 1
        except Exception as e:
            print(f"Failed to migrate {db_id}: {e}")
            
    print(f"Successfully migrated {count} databases to {OUTPUT_DIR}")
    print("These JSON files can be loaded into MongoDB or mongomock.")

if __name__ == "__main__":
    main()
