import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.rag import initialize_retriever, retrieve_schema

def main() -> None:
    """
    Test script to verify RAG retrieval logic without loading the full model.
    """
    schemas = [
        "CREATE TABLE users (id INT, name TEXT, age INT, email TEXT);",
        "CREATE TABLE orders (id INT, user_id INT, amount DECIMAL, order_date DATE);",
        "CREATE TABLE products (id INT, name TEXT, price DECIMAL, stock INT);"
    ]
    
    print("Initializing retriever...")
    initialize_retriever(schemas)
    
    query = "Show me all users who are older than 25"
    print(f"\nQuery: {query}")
    retrieved = retrieve_schema(query)
    print(f"Retrieved Schema:\n{retrieved}")
    
    if "users" in retrieved:
        print("\nSUCCESS: Retrieved correct schema.")
    else:
        print("\nFAILURE: Did not retrieve correct schema.")

if __name__ == "__main__":
    main()
