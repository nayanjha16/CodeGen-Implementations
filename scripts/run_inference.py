import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.model_loader import load_model
from src.inference import generate_sql
from src.rag import retrieve_schema, initialize_retriever

# Dummy schemas for demonstration
DUMMY_SCHEMAS = [
    "CREATE TABLE users (id INT, name TEXT, age INT, email TEXT);",
    "CREATE TABLE orders (id INT, user_id INT, amount DECIMAL, order_date DATE);",
    "CREATE TABLE products (id INT, name TEXT, price DECIMAL, stock INT);"
]

def main() -> None:
    """
    Main entry point for the interactive Text-to-SQL inference application.
    """
    print("Initializing Text-to-SQL Inference...")
    
    # Initialize RAG
    initialize_retriever(DUMMY_SCHEMAS)
    
    model, tokenizer = load_model()
    
    print("\nModel loaded. Ready for queries.")
    print("Example query: 'Show me all users who are older than 25'")
    
    while True:
        query = input("\nEnter your query (or 'exit' to quit): ")
        if query.lower() == 'exit':
            break
            
        # Retrieve relevant schema
        schema = retrieve_schema(query)
        print(f"\n[Retrieved Context]:\n{schema}\n")
        
        sql = generate_sql(model, tokenizer, query, schema)
        print(f"Generated SQL:\n{sql}")

if __name__ == "__main__":
    main()
