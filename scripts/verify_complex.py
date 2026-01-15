import sys
import os

sys.path.append(os.getcwd())

from src.inference import generate_nosql, generate_sql
# Mock model/tokenizer for unit testing logic (not end-to-end LLM in this script unless we load it, 
# but here we test the deterministic SQL->NoSQL mostly)

def test_complex_conversion():
    queries = [
        # 1. HAVING clause
        "SELECT department, avg(salary) FROM employees GROUP BY department HAVING avg(salary) > 50000",
        
        # 2. Subquery (IN) - Note: Will fail execution if DB not present, but should generate structure matching IDs
        # We simulate DB path being None for structure check, or we could mock sqlite
        "SELECT name FROM users WHERE id IN (SELECT user_id FROM orders)",
        
        # 3. Complex JOIN + Filter
        "SELECT t1.name, t2.total FROM t1 JOIN t2 ON t1.id = t2.t1_id WHERE t2.total > 100",
        
        # 4. Limit + Sort
        "SELECT * FROM products ORDER BY price DESC LIMIT 3"
    ]
    
    print("=== Testing Complex SQL to NoSQL Conversion ===\n")
    
    for i, sql in enumerate(queries):
        print(f"--- Complex Test Case {i+1} ---")
        print(f"SQL: {sql}")
        try:
            # Pass None for db_path to skip actual execution but test parsing/logic flow
            # For subquery, it won't inject IDs but should show structure
            nosql = generate_nosql(None, None, sql, None, db_path=None) 
            print(f"MongoDB:\n{nosql}\n")
        except Exception as e:
            print(f"ERROR: {e}\n")

if __name__ == "__main__":
    test_complex_conversion()
