import json
import os
import sqlite3
import pymongo
import openai
from collections import defaultdict
from tqdm import tqdm
from func_timeout import func_timeout, FunctionTimedOut

# --- CONFIGURATION (Edit these paths if needed) ---
OPENAI_API_KEY = "Your API Key"  # <--- PASTE KEY HERE

# Relative paths assume you run this from the TextToNoSQL folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_TRAIN_FILE = os.path.join(BASE_DIR, "train.json")
SQL_DIR = os.path.join(BASE_DIR, "train_sqlite")
NOSQL_DIR = os.path.join(BASE_DIR, "train_nosql")
OUTPUT_FILE = os.path.join(BASE_DIR, "train_nosql.json")

# Connect to Local Services
client = openai.OpenAI(api_key=OPENAI_API_KEY)
mongo_client = pymongo.MongoClient("mongodb://localhost:27017/")

# --- HELPER: Load DB into Mongo ---
def load_db_into_mongo(db_id):
    """Imports the JSON file into a MongoDB database named after db_id."""
    # check for filename variants (e.g. "concert_singer.json" or "train_concert_singer.json")
    candidates = [
        os.path.join(NOSQL_DIR, f"{db_id}.json"),
        os.path.join(NOSQL_DIR, f"train_{db_id}.json")
    
    ]
    
    json_path = None
    for c in candidates:
        if os.path.exists(c):
            json_path = c
            break
    
    if not json_path:
        # Silent fail to keep logs clean, or print warning
        return False

    db = mongo_client[db_id]
    
    # Skip if already loaded to save time/RAM
    if len(db.list_collection_names()) > 0:
        return True

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        for coll_name, docs in data.items():
            if docs:
                db[coll_name].insert_many(docs)
        return True
    except Exception as e:
        print(f"\n[Error] Failed to import {db_id}: {e}")
        return False

# --- HELPER: CoT Pipeline ---
def run_pipeline(db_id, nlq, evidence, gold_sql):
    db = mongo_client[db_id]
    
    # 1. Get Schema
    schema = {}
    for coll in db.list_collection_names():
        if "system" not in coll:
            doc = db[coll].find_one()
            if doc: schema[coll] = [k for k in doc.keys() if k != '_id']
            
    # 2. Generate

    # We explicitly tell it to use the keys from the schema dictionary
    collection_names = list(schema.keys())
    
    prompt = f"""
    You are a PyMongo expert. Convert the Natural Language Query (NLQ) to a SINGLE Python expression.
    
    ## Database Schema:
    {json.dumps(schema, indent=2)}
    
    ## NLQ:
    "{nlq}"
    
    ## CRITICAL INSTRUCTIONS:
    1. Output ONLY the query expression starting with 'db'.
    2. DO NOT import pymongo. DO NOT create a MongoClient. DO NOT use print().
    3. DO NOT assign the result to a variable (e.g., do NOT write 'result = ...').
    4. The code will be executed as: `res = <YOUR_OUTPUT_HERE>`
    5. Use case-insensitive regex for string matching: {{"$regex": "pattern", "$options": "i"}}
    
    ## ONE-SHOT LEARNING EXAMPLE (Flat/Relational Schema):
    # This example teaches how to JOIN separate collections using $lookup    
    # NLQ: "What is the total number of households in Arecibo county?"
    # Output:
    db.zip_data.aggregate([
        {{"$lookup": {{
            "from": "country",              # Join with 'country' collection
            "localField": "zip_code",       # Field in 'zip_data'
            "foreignField": "zip_code",     # Field in 'country'
            "as": "country_info"            # Result alias
        }}}},
        {{"$unwind": "$country_info"}},     # Flatten the joined array to filter it
        {{"$match": {{"country_info.county": "ARECIBO"}}}}, 
        {{"$group": {{"_id": null, "total": {{"$sum": "$households"}}}}}}
    ])
    # "Note: If the join key is an integer in one table and string in another, use $toString or $toInt in $addFields before looking up."
    6. Use one of these collection names: {collection_names}
    
    A: Let's think step by step!
    """
    try:
        res = client.chat.completions.create(
            model="gpt-4o", messages=[{"role": "user", "content": prompt}], temperature=0
        )
        nosql_query = extract_code(res.choices[0].message.content)
    except Exception as e:
        # ADDING THIS LINE TO SEE THE ERROR:
        print(f"OPENAI ERROR: {e}")
        return {"error": str(e), "nosql_query": "GEN_ERROR", "is_correct": False}

    # 3. Verify
    sql_path = os.path.join(SQL_DIR, f"{db_id}.sqlite")
    if not os.path.exists(sql_path):
        return {"error": "Missing SQLite file", "nosql_query": nosql_query, "is_correct": False}

    # Execute SQL
    sql_res = "SQL_ERROR"
    conn = sqlite3.connect(sql_path)
    try:
        cursor = conn.cursor()
        cursor.execute(gold_sql)
        sql_res = [list(row) for row in cursor.fetchall()]
        sql_res.sort(key=lambda x: str(x))
    except:
        pass
    finally:
        conn.close()

    # Execute NoSQL
    is_correct = False
    try:
        #local_env = {"db": db}

        # Ensuring the code is an expression we can assign
        # We strip any trailing semi-colons which are common in SQL habits
        clean_query = nosql_query.strip().rstrip(';')
        if "=" in clean_query: 
             clean_query = clean_query.split("=", 1)[1].strip()
        
        # ADDING  DEBUG PRINT:
        #print(f"   ▶ Executing: {nlq[:50]}...")
        
        # RUNNING WITH 5-SECOND TIMEOUT
        try:
            nosql_res = func_timeout(5, safe_exec, args=(clean_query, db))
        except FunctionTimedOut:
            print(f"   ⏳ TIMEOUT: Query took too long. Skipping.")
            return {"error": "TIMEOUT", "nosql_query": nosql_query, "verified": False}
        except Exception as e:
            # print(f"Exec Error: {e}") 
            pass
             
        # Execute
        # exec(f"res = {clean_query}", {}, local_env)
        # nosql_res = local_env.get('res')                   
        
        # 1. Convert PyMongo Cursor to List
        if hasattr(nosql_res, 'close') and hasattr(nosql_res, '__iter__'): # It's a cursor
            #nosql_res = list(nosql_res)
            # NEW: Only take first 100 rows to prevent RAM crash
             nosql_res = [doc for _, doc in zip(range(100), nosql_res)]
        if not isinstance(nosql_res, list): # It's a single value (int/float)
            nosql_res = [nosql_res]
            
        # 2. Flatten Dictionaries & Remove '_id'
        cleaned_nosql = []
        for item in nosql_res:
            if isinstance(item, dict):                
                item.pop('_id', None)  # Remove _id if present                
                cleaned_nosql.append(list(item.values()))  # Extract just the values (e.g., {"total": 1500} -> [1500])
            elif isinstance(item, (list, tuple)):
                cleaned_nosql.append(list(item))
            else:
                cleaned_nosql.append([item])
        
        # 3. Sort for Set Comparison (ignore order)
        # Convert all elements to strings to handle mixed types safely  
        nosql_res_sorted = sorted(nosql_res, key=lambda x: str(x)) 
        sql_res_sorted = sorted(sql_res, key=lambda x: str(x)) if isinstance(sql_res, list) else sql_res        
        
        if str(sql_res_sorted) == str(nosql_res_sorted):
            is_correct = True
        # else:
        #     # Debugging:
        #     print(f"❌ MISMATCH for: {nlq}")
        #     print(f"   SQL Expected: {str(sql_res_sorted)[:100]}")  # Print first 100 chars 
        #     print(f"   NoSQL Got:    {str(nosql_res_sorted)[:100]}") # Print first 100 chars            
        #     print(f"   Query Used:   {clean_query}\n")
            
    except Exception:
        pass

    return {"nosql_query": nosql_query, "is_correct": is_correct}

def safe_exec(query_str, db_obj):
    """Runs exec() in a way that returns the result."""
    local_env = {"db": db_obj}
    exec(f"res = {query_str}", {}, local_env)
    return local_env.get('res')

def extract_code(text):
    # Strip markdown code blocks
    if "```" in text:
        text = text.split("```")[1].replace("python", "").strip()
    # Clean up common LLM mistakes
    text = text.strip()
    # Remove imports if present
    lines = text.split('\n')
    clean_lines = []
    for line in lines:
        if "import " in line or "MongoClient" in line or "print(" in line:
            continue
        # Remove variable assignment (e.g., "res = db..." -> "db...")
        if "=" in line and "db." in line.split("=")[1]:
            line = line.split("=")[1].strip()
        clean_lines.append(line)
        
    return "\n".join(clean_lines).strip()

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    print(f"Reading {INPUT_TRAIN_FILE}...")
    try:
        with open(INPUT_TRAIN_FILE, 'r') as f:
            questions = json.load(f)
    except FileNotFoundError:
        print("Error: train.json not found. Please put it in the same folder as this script.")
        exit()

    # Group by DB
    tasks = defaultdict(list)
    for q in questions:
        if 'db_id' in q: tasks[q['db_id']].append(q)

    results = []
    
    print(f"Found {len(questions)} questions across {len(tasks)} databases.")
    print("Starting processing... (Press Ctrl+C to stop safely)")

    try:
        # Progress bar for Databases
        for db_id, items in tqdm(tasks.items(), desc="Databases"):
            
            # Load Data (Only once per DB)
            if not load_db_into_mongo(db_id):
                continue
            
            # Process Questions
            for item in items:
                res = run_pipeline(db_id, item['question'], item['evidence'], item['SQL'])
                
                results.append({
                    "db_id": db_id,
                    "question": item['question'],
                    "SQL": item['SQL'],
                    "generated_nosql": res['nosql_query'],
                    "verified": res.get('is_correct', False)
                })
                
    except KeyboardInterrupt:
        print("\nStopping early...")

    # Save results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDone! Saved {len(results)} results to {OUTPUT_FILE}")