import sys
import os
import torch

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_loader import load_model
from src.inference import generate_sql
from src.dataset_loader import load_spider_tables, get_database_schema
from src.rag import initialize_retriever, retrieve_schema
import src.config as config

def main():
    print("Loading resources...")
    print("Loading model (this might take a moment)...")
    # Load model (defaults to baseline config)
    model, tokenizer = load_model()
    
    # Load schemas for RAG
    print("Loading schemas from tables.json...")
    try:
        tables_dict = load_spider_tables()
    except Exception as e:
        print(f"Error loading tables: {e}")
        return

    schemas = []
    # We need a list of schema strings. 
    # get_database_schema(db_id) returns the detailed CREATE TABLE string.
    print("Preparing schema corpus...")
    failed_dbs = 0
    for db_id in tables_dict.keys():
        try:
            schema = get_database_schema(db_id)
            schemas.append(schema)
        except Exception:
            failed_dbs += 1
            
    print(f"Loaded {len(schemas)} schemas. ({failed_dbs} failed)")
    
    if not schemas:
        print("Error: No schemas loaded. Check data.")
        return

    # Initialize RAG
    # Note: initialization might take a moment to encode all schemas
    initialize_retriever(schemas)
    
    print("\n" + "="*50)
    print("Text-to-SQL Inference")
    print("Type 'quit' or 'exit' to stop.")
    print("="*50)
    
    while True:
        try:
            query = input("\nEnter your question: ")
            if query.strip().lower() in ['quit', 'exit']:
                break
            
            if not query.strip():
                continue
                
            print(f"Retrieving schema context...")
            schema_context = retrieve_schema(query)
            
            print("Generating SQL...")
            sql = generate_sql(model, tokenizer, query, schema_context)
            
            print("\nGenerated SQL:")
            print("-" * 20)
            print(sql)
            print("-" * 20)
            
        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
