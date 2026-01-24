
import os
import json
import time
import argparse
import google.generativeai as genai
from tqdm import tqdm
from typing import List, Dict

import sys
# Add project root to path so we can import 'services' module
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

# Reuse the prompt template from our existing codebase
try:
    from services.inference.prompts import SQL_TO_MQL_SYSTEM_PROMPT
except ImportError as e:
    print(f"CRITICAL ERROR: Could not import prompts.py: {e}")
    print(f"Python Path: {sys.path}")
    sys.exit(1)

def load_data(filepath: str) -> List[Dict]:
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_data(data: List[Dict], filepath: str):
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def convert_with_gemini(sql_query: str, db_id: str, model_name: str = "gemini-1.5-flash") -> Dict:
    """Uses Gemini to convert SQL to MQL."""
    
    # Simple schema representation (in real training we might want full schema but for now this is efficient)
    # The prompt actually expects a schema map.
    # For bulk conversion without full DB access, we can either:
    # 1. Load the actual DB schema scanning (complex)
    # 2. Let Gemini infer from SQL column names (heuristically works well for conversion)
    # 3. Pass a placeholder or minimal context
    
    # Strategy: Using inferred schema from SQL to keep cost low and speed high
    # The model is smart enough to understand the structure from the SQL itself usually.
    
    fake_schema = f"Schema for {db_id} (inferred from query)" 
    
    prompt = SQL_TO_MQL_SYSTEM_PROMPT.format(
        sql_query=sql_query,
        mongo_schema=fake_schema 
    ) + "\n\nIMPORTANT: Return ONLY the JSON array. Do not wrap in markdown blocks."

    max_retries = 5
    retry_delay = 10
    
    for attempt in range(max_retries):
        try:
            model = genai.GenerativeModel(model_name)
            response = model.generate_content(prompt)
            
            text_response = response.text.replace("```json", "").replace("```", "").strip()
            
            try:
                mql = json.loads(text_response)
                return mql
            except json.JSONDecodeError:
                print(f"  [WARN] Invalid JSON for SQL: {sql_query[:50]}...")
                return None
                
        except Exception as e:
            if "Quota exceeded" in str(e) or "429" in str(e):
                wait_time = retry_delay * (2 ** attempt) # Exponential backoff: 10, 20, 40...
                print(f"  [WARN] Rate limit hit. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                time.sleep(wait_time)
            else:
                print(f"  [ERROR] Gemini API Error: {e}")
                return None
    return None

def main():
    parser = argparse.ArgumentParser(description="Convert SQL to MQL using Gemini")
    parser.add_argument("--input", default="data/training/bird_train.json", help="Input JSON file")
    parser.add_argument("--output", default="data/training/bird_train_with_mql.json", help="Output JSON file")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of examples (0 for all)")
    parser.add_argument("--model", default="gemini-flash-latest", help="Gemini model to use")
    parser.add_argument("--api_key", required=False, help="Google API Key (or set GOOGLE_API_KEY env var)")
    parser.add_argument("--sleep", type=float, default=10.0, help="Sleep time between requests in seconds")
    
    args = parser.parse_args()
    
    # Setup API Key
    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: Authorization failed. Please provide --api_key or set GOOGLE_API_KEY environment variable.")
        return
        
    genai.configure(api_key=api_key)
    
    # Load Data
    data = load_data(args.input)
    print(f"Loaded {len(data)} examples from {args.input}")
    
    # Filter processed
    if os.path.exists(args.output):
        existing_data = load_data(args.output)
        processed_count = len([x for x in existing_data if "mql_pipeline" in x])
        print(f"Found existing output with {processed_count} processed examples. Resuming...")
        # Smart Resume: Map existing results by SQL or ID to source data
        # Simply replacing data with existing won't work if inputs are split files
        # For split files, just assume we process what's given.
        # But if output exists, we should probably load it to check completion.
        # simpler approach for split files: just load output if exists, and skip items present in it
        
        existing_map = {json.dumps(x['sql']): x.get('mql_pipeline') for x in existing_data if 'mql_pipeline' in x}
        for item in data:
            s = json.dumps(item['sql'])
            if s in existing_map:
                item['mql_pipeline'] = existing_map[s]

    # Limit if requested
    target_data = data
    if args.limit > 0:
        target_data = data[:args.limit]
        print(f"Limiting to first {args.limit} examples.")
        
    success_count = 0
    
    print(f"Starting conversion using {args.model}...")
    print(f"Rate Limit Protection: Sleeping {args.sleep} seconds between requests...")
    
    for i, item in enumerate(tqdm(target_data)):
        if "mql_pipeline" in item:
            continue # Skip already processed
            
        mql = convert_with_gemini(item["sql"], item["db_id"], args.model)
        
        if mql:
            item["mql_pipeline"] = mql
            success_count += 1
            
            # Save checkpoint every 10 items
            if success_count % 10 == 0:
                save_data(target_data, args.output)
        
        # Respect Rate Limit
        time.sleep(args.sleep) 
        
    # Final Save
    save_data(target_data, args.output)
    print(f"Done! converted {success_count} new examples. Saved to {args.output}")

if __name__ == "__main__":
    main()
