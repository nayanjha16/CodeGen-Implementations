
import json
import argparse
import random

# Use the EXACT System Prompt used during training/inference
SYSTEM_PROMPT = """You are a MongoDB Expert. Convert the following SQLite query into a MongoDB Aggregation Pipeline.

Input SQL: {sql_query}

Target Schema: {mongo_schema}

Rules:
1. Use the $lookup stage for all JOINs.
2. Immediately follow $lookup with $unwind if the SQL implies a 1:1 or Inner Join relationship.
3. For COUNT(*), use [{"$count": "count"}]
4. For simple SELECT *, use [{"$match": {}}] to return all documents
5. Return ONLY a valid JSON array. No markdown, no explanations.
"""

import sqlite3
import os

def load_schema_map_from_json(json_path):
    """
    Parses train_tables.json to build a map of db_id -> schema_string.
    """
    if not os.path.exists(json_path):
        print(f"Schema JSON not found at {json_path}")
        return {}
        
    print(f"Loading schema definitions from {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        tables_data = json.load(f)
        
    schema_map = {}
    
    for db in tables_data:
        db_id = db["db_id"]
        table_names = db["table_names_original"]
        col_names = db["column_names_original"]
        col_types = db["column_types"]
        foreign_keys = db["foreign_keys"]
        
        # Organize columns by table index
        # col_names is list of [table_idx, col_name]
        # table_idx -1 is special (*)
        
        table_cols = {i: [] for i in range(len(table_names))}
        
        for idx, (tbl_idx, col_name) in enumerate(col_names):
            if tbl_idx == -1: continue
            col_type = col_types[idx]
            table_cols[tbl_idx].append(f"{col_name} ({col_type})")
            
        parts = []
        for i, tbl_name in enumerate(table_names):
            cols_str = ", ".join(table_cols[i])
            parts.append(f"Table: {tbl_name}\nColumns: {cols_str}")
            
        # Foreign Keys: [col_idx_1, col_idx_2]
        for fk in foreign_keys:
            c1_idx, c2_idx = fk
            
            # Find table and col name for c1
            t1_idx, c1_name = col_names[c1_idx]
            t1_name = table_names[t1_idx]
            
            # Find table and col name for c2
            t2_idx, c2_name = col_names[c2_idx]
            t2_name = table_names[t2_idx]
            
            parts.append(f"Relationship: {t1_name}.{c1_name} -> {t2_name}.{c2_name}")
            
        schema_map[db_id] = "\n\n".join(parts)
        
    return schema_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/training/bird_train_with_mql.json")
    parser.add_argument("--output", default="data/training/sql_to_mql_finetuning.jsonl")
    parser.add_argument("--schema_json", default="data/training/train_tables.json")
    args = parser.parse_args()
    
    print(f"Loading {args.input}...")
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    # Load Schema from JSON
    schema_cache = load_schema_map_from_json(args.schema_json)
    print(f"Loaded definitions for {len(schema_cache)} databases.")
            
    valid_count = 0
    missing_schema_count = 0
    
    with open(args.output, 'w', encoding='utf-8') as out_f:
        for item in data:
            if "mql_pipeline" not in item:
                continue
                
            sql = item["sql"]
            mql = json.dumps(item["mql_pipeline"])
            db_id = item["db_id"]
            
            # Fetch Schema
            if db_id in schema_cache:
                schema_context = schema_cache[db_id]
            else:
                schema_context = "Schema information not available. Infer structure from input SQL."
                missing_schema_count += 1
            
            # Construct Prompt Content
            user_content = f"Input SQL: {sql}\n\nTarget Schema: {schema_context}"
            
            messages = [
                {"role": "system", "content": "You are a MongoDB Expert. Convert SQL to MQL."},
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": mql}
            ]
            
            out_f.write(json.dumps({"messages": messages}) + "\n")
            valid_count += 1

            
    print(f"Done. Saved {valid_count} formatted examples to {args.output}")
    if missing_schema_count > 0:
        print(f"WARNING: {missing_schema_count} examples were missing schema context (DB ID not in tables.json).")

if __name__ == "__main__":
    main()
