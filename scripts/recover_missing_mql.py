import json
import os
import time
import argparse
import google.generativeai as genai
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import sys

# Add path to import prompts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from services.inference.prompts import SQL_TO_MQL_SYSTEM_PROMPT
except ImportError:
    # Fallback if path issue
    SQL_TO_MQL_SYSTEM_PROMPT = "You are a MongoDB Expert. Convert the following SQL query to a MongoDB Aggregation Pipeline (MQL). Return ONLY the JSON pipeline in a list [ ... ]. Do not include explanations. SQL: {sql} \n Schema: {schema}"

def load_api_keys(key_file):
    with open(key_file, 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def convert_item(item, model, schema_map):
    if "mql_pipeline" in item:
        return item
        
    sql = item['sql']
    db_id = item['db_id']
    schema_context = schema_map.get(db_id, "Schema not available.")
    
    prompt = f"Target Schema: {schema_context}\n\nSQL: {sql}"
    
    try:
        response = model.generate_content(
            [SQL_TO_MQL_SYSTEM_PROMPT, prompt],
            generation_config={"response_mime_type": "application/json"}
        )
        mql = json.loads(response.text)
        item['mql_pipeline'] = mql
        return item
    except Exception as e:
        # print(f"Error converting {sql[:50]}...: {e}")
        return item # Return original without mql_pipeline to retry later? Or just keep it.

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/training/bird_train_with_mql.json")
    parser.add_argument("--schema", default="data/training/train_tables.json")
    parser.add_argument("--api_keys", default="api_keys.txt")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    # Load Data
    print(f"Loading data from {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Load Schema
    from prepare_finetuning_data import load_schema_map_from_json
    schema_map = load_schema_map_from_json(args.schema)
    
    # Identify items to process
    missing_indices = [i for i, item in enumerate(data) if "mql_pipeline" not in item]
    print(f"Found {len(missing_indices)} items missing MQL pipeline.")
    
    if not missing_indices:
        print("Nothing to do.")
        return

    # Setup Gemini
    keys = load_api_keys(args.api_keys)
    if not keys:
        print("No API keys found!")
        return
        
    # Round-robin keys
    print(f"Using {len(keys)} API keys.")
    
    # Process
    # We will process in chunks and save progressively
    
    # Just use one model instance per worker? No, model is tied to key.
    # We'll create a pool of model clients?
    # Actually, simpler to just randomly select key per request or round robin.
    
    import random
    
    def worker_func(idx):
        item = data[idx]
        key = random.choice(keys)
        genai.configure(api_key=key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Retry loop
        for attempt in range(3):
            try:
                converted = convert_item(item, model, schema_map)
                if "mql_pipeline" in converted:
                    return idx, converted
            except Exception as e:
                time.sleep(1 * (attempt + 1))
        return idx, item # Failed

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker_func, i): i for i in missing_indices}
        
        success_count = 0
        for future in tqdm(as_completed(futures), total=len(missing_indices)):
            idx, result_item = future.result()
            data[idx] = result_item
            if "mql_pipeline" in result_item:
                success_count += 1
            
            # Save every 500?
            if success_count % 500 == 0:
                 with open(args.input, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                    
    # Final Save
    print(f"Finished. Converted {success_count} new items.")
    with open(args.input, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    main()
