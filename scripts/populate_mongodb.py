
import os
import glob
from pathlib import Path
from setup_mongodb import get_mongo_client

def populate_mongodb():
    client = get_mongo_client()
    db = client["bird_nosql"]
    
    scripts_dir = Path("data/bird/nosql_scripts")
    if not scripts_dir.exists():
        print("No NoSQL scripts directory found.")
        return

    files = list(scripts_dir.glob("*.js"))
    print(f"Found {len(files)} NoSQL scripts to execute.")
    
    for js_file in files:
        collection_name = js_file.stem.replace("_nosql", "")
        # The file contains raw JS "db.collection.insertMany(...)". 
        # Executing raw JS on mongo via python is tricky without 'eval' or parsing.
        # Since we generated valid JSON-like structures in the JS, a better approach for this demo 
        # might have been generating JSON files.
        # However, to support the "script" requirement, we'll try to parse the JSON array out of the JS text.
        
        print(f"Populating collection: {collection_name}")
        

        try:
            with open(js_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Regex to find all insertMany calls: db.collection.insertMany([...]);
            # We look for the JSON array inside. 
            import re
            
            # Pattern: insertMany\((.*?)\);
            # Note: JSON inside might be complex, so regex is fragile if nested parens exist.
            # But our generator produces `insertMany([ ... ]);` cleanly per batch.
            # A safer way strictly for our generated format is to look for "insertMany(" and the closing ");"
            # Or since we know we write one call per line (roughly) or per block.
            
            # Let's iterate over the file content using a split or simple find
            matches = re.finditer(r'db\.[a-zA-Z0-9_]+\.insertMany\((.*?)\);', content, re.DOTALL)
            
            count = 0
            for match in matches:
                json_str = match.group(1)
                import json
                try:
                    data = json.loads(json_str)
                    if data:
                        # For the first batch, maybe drop? But we have multiple batches.
                        # We should drop ONLY ONCE at the start.
                        if count == 0 and collection_name in db.list_collection_names():
                             # Only drop if we are sure this is a new run? 
                             # Risk: If we have multiple files for same collection (unlikely here), we overwrite.
                             # For now, let's just insert. If user runs twice, data dupes. 
                             # Let's try to drop if it exists and this is the first batch processed for this file.
                             db[collection_name].drop()
                             
                        db[collection_name].insert_many(data)
                        count += len(data)
                except json.JSONDecodeError as e:
                     print(f" - Failed to parse batch: {e}")
            
            if count > 0:
                print(f" - Populated {collection_name} with {count} documents.")
            else:
                print(f" - No data found in {js_file.name}")
                
        except Exception as e:
            print(f"Error processing {js_file}: {e}")

if __name__ == "__main__":
    populate_mongodb()
