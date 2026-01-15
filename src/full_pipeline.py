
import os
import sys
# Add project root to sys.path to allow imports from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from model_loader import load_model
from rag import retrieve_schema, initialize_retriever
from inference import generate_sql, generate_nosql
from dataset_loader import load_bird_tables

def main():
    print("Initializing Text-to-SQL-to-NoSQL Pipeline...")
    
    # 1. Setup RAG
    # For demo, we load BirdBench tables
    tables = load_bird_tables()
    if not tables:
        print("No BirdBench tables found. RAG will be empty.")
        schemas = []
    else:
        schemas = []
        for db_id, info in tables.items():
            # Construct a simple schema text representation
            # This is a naive dump; real RAG needs better formatting
            text_repr = f"Database: {db_id}\n"
            # ... (formatting logic same as get_bird_schema roughly)
            schemas.append({'text': text_repr, 'id': db_id}) # Placeholder logic
            
        initialize_retriever(schemas)

    # 2. Load Model
    # Using one model for both steps for now (0.5B)
    model, tokenizer = load_model()
    
    # 3. Interactive Loop
    # 3. Interactive Loop
    while True:
        print("\n" + "="*40)
        print("Select Mode:")
        print("1. Text to SQL (Input: Natural Language -> Output: SQL)")
        print("2. SQL to NoSQL (Input: SQL -> Output: MongoDB Query)")
        print("q. Quit")
        print("="*40)
        
        mode = input("Choice: ").strip().lower()
        if mode in ['q', 'exit', 'quit']: break
        
        if mode == '1':
            text = input("\n[Text-to-SQL] Enter Question: ")
            if text.lower() in ['q', 'exit', 'quit']: break
            
            print(" [1] Retrieving Schema...")
            schema_context, db_id = retrieve_schema(text)
            
            print(" [2] Generating SQL...")
            sql = generate_sql(model, tokenizer, text, schema_context)
            print(f"\nGenerated SQL:\n{sql}")
            
        elif mode == '2':
            sql_input = input("\n[SQL-to-NoSQL] Enter SQL Query: ")
            if sql_input.lower() in ['q', 'exit', 'quit']: break
            
            # We still try to retrieve schema context using the SQL to find relevant tables
            print(" [1] Retrieving Schema Context...")
            schema_context, db_id = retrieve_schema(sql_input)
            
            print(" [2] Converting to NoSQL...")
            nosql = generate_nosql(model, tokenizer, sql_input, schema_context)
            print(f"\nGenerated NoSQL:\n{nosql}")
            
        else:
            print("Invalid choice. Please enter 1, 2, or q.")

if __name__ == "__main__":
    main()
