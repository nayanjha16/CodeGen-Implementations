from flask import Flask, render_template, request, jsonify
import sys
import os
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.model_loader import load_model
from src.inference import generate_sql, generate_nosql
from src.dataset_loader import load_spider_tables, get_database_schema, get_unified_schema
from src.rag import initialize_retriever, retrieve_schema
from src.nosql_utils import load_sqlite_to_mongo, execute_mongo_query, compare_results, load_sqlite_to_mongo
import src.config as config
import sqlite3

app = Flask(__name__)

# Global variables to hold model and schemas
model = None
tokenizer = None
schemas_loaded = False

def initialize_app():
    global model, tokenizer, schemas_loaded
    if model is None:
        print("Loading model for web app...")
        model, tokenizer = load_model()
    
    if not schemas_loaded:
        print("Loading schemas for RAG...")
        try:
            tables_dict = load_spider_tables()
            schemas = []
            for db_id in tables_dict.keys():
                try:
                    schema_text = get_unified_schema(db_id)
                    schemas.append({'text': schema_text, 'id': db_id})
                except Exception:
                    pass
            
            if schemas:
                initialize_retriever(schemas)
                schemas_loaded = True
                print(f"RAG initialized with {len(schemas)} schemas.")
            else:
                print("Warning: No schemas loaded.")
        except Exception as e:
            print(f"Error loading schemas: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/generate', methods=['POST'])
def generate():
    global model, tokenizer, schemas_loaded
    
    # Ensure initialized on first request if not already
    if model is None:
        initialize_app()
        
    data = request.json
    query = data.get('query')
    mode = data.get('mode', 'text') # 'text' or 'sql'
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
        
    try:
        # Retrieve schema
        schema_context, db_id = retrieve_schema(query)
        
        # Generator SQL
        if mode == 'sql':
            sql = query # Input is already SQL
        else:
            sql = generate_sql(model, tokenizer, query, schema_context)
        
        # Generate NoSQL
        # Determine DB Path first (needed for subquery execution in NoSQL)
        db_path = None
        if db_id:
             p1 = os.path.join(config.DATA_DIR, 'spider', 'database', db_id, f'{db_id}.sqlite')
             p2 = os.path.join(config.DATA_DIR, 'bird', 'databases', db_id, f'{db_id}.sqlite')
             p3 = f"data/bird/minidev/MINIDEV/dev_databases/{db_id}/{db_id}.sqlite"
             
             if os.path.exists(p1): db_path = p1
             elif os.path.exists(p2): db_path = p2
             elif os.path.exists(p3): db_path = p3

        # Generate NoSQL (Pass db_path for subqueries)
        nosql = generate_nosql(model, tokenizer, sql, schema_context, db_path=db_path)
        
        result_data = {
            'sql': sql,
            'nosql': nosql,
            'db_id': db_id,
            'execution_match': None,
            'sql_result': None,
            'nosql_result': None,
            'error': None
        }
        
        # Execution (if DB available)
        if db_id:
            # Construct DB path (Assuming Spider structure default)
            db_path = os.path.join(config.DATA_DIR, 'spider', 'database', db_id, f'{db_id}.sqlite')
            if not os.path.exists(db_path):
                # Try Bird path
                # Try Bird path
                db_path = os.path.join(config.DATA_DIR, 'bird', 'databases', db_id, f'{db_id}.sqlite')
                if not os.path.exists(db_path):
                     # Try Bird Mini-Dev path
                     db_path = f"data/bird/minidev/MINIDEV/dev_databases/{db_id}/{db_id}.sqlite"
            
            if os.path.exists(db_path):
                # 1. Execute SQL
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute(sql)
                    sql_res = cursor.fetchall()
                    conn.close()
                    result_data['sql_result'] = str(sql_res[:5]) # Sample
                except Exception as e:
                    result_data['error'] = f"SQL Execution Error: {e}"
                    sql_res = None
                
                # 2. Execute NoSQL
                if sql_res is not None:
                    try:
                        # Load Mock Data
                        mongo_client = load_sqlite_to_mongo(db_path)
                        mongo_res, err = execute_mongo_query(mongo_client, nosql)
                        
                        if err:
                            result_data['nosql_error'] = err
                        else:
                            result_data['nosql_result'] = str(mongo_res[:5]) # Sample
                            # 3. Compare
                            is_match = compare_results(sql_res, mongo_res)
                            result_data['execution_match'] = is_match
                    except Exception as e:
                        result_data['nosql_error'] = f"NoSQL Error: {e}"
            else:
                 result_data['error'] = f"Database file not found for {db_id}"
        
        return jsonify(result_data)
        
    except Exception as e:
        print(f"Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Initialize before running server to avoid timeout on first request
    initialize_app()
    app.run(debug=True, use_reloader=False) 
