
import sqlite3
import os

DB_PATH = r"c:\2025-AI\Capstone-Project\IIT-Project\release-5.0-V1-Kumar\text-to-SQL\data\bird\minidev\MINIDEV\dev_databases\financial\financial.sqlite"

def run_query():
    if not os.path.exists(DB_PATH):
        print(f"Error: Database not found at {DB_PATH}")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # 1. Check schemas
        print("--- Tables in Database ---")
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        for t in tables:
            print(f"- {t[0]}")
            
        # 2. Try to match table name from user query "financial_transactions"
        # Since I am not 100% sure if financial_transactions exists, I will query columns if it does.
        
        user_sql = """
        SELECT account_id, SUM(amount) AS total_amount
        FROM trans
        WHERE date >= '1997-12-31'
        GROUP BY account_id
        ORDER BY total_amount DESC
        LIMIT 10;
        """
        
        print("\n--- Executing Query ---")
        print(user_sql)
        cursor.execute(user_sql)
        results = cursor.fetchall()
        
        print("\n--- Results ---")
        print(f"{'Customer ID':<15} | {'Total Amount'}")
        print("-" * 30)
        for row in results:
            print(f"{row[0]:<15} | {row[1]}")
            
        conn.close()
        
    except Exception as e:
        print(f"\nError Executing Query: {e}")

if __name__ == "__main__":
    run_query()
