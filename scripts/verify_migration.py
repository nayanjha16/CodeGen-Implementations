import sys
import os

# Add root to string
sys.path.append(os.getcwd())

from src.inference import generate_nosql

def test_conversion():
    queries = [
        # 1. Simple Select
        "SELECT name, age FROM users WHERE age > 25",
        
        # 2. Aggregation
        "SELECT department, avg(salary) FROM employees GROUP BY department",
        
        # 3. Join
        "SELECT orders.id, customers.name FROM orders JOIN customers ON orders.customer_id = customers.id WHERE orders.amount > 100",
        
        # 4. Sort and Limit
        "SELECT * FROM products ORDER BY price DESC LIMIT 5"
    ]
    
    print("=== Testing SQL to NoSQL Conversion (Deterministic) ===\n")
    
    for i, sql in enumerate(queries):
        print(f"--- Test Case {i+1} ---")
        print(f"SQL: {sql}")
        try:
            nosql = generate_nosql(None, None, sql, None)
            print(f"MongoDB:\n{nosql}\n")
        except Exception as e:
            print(f"ERROR: {e}\n")

if __name__ == "__main__":
    test_conversion()
