import sqlite3
import json
from pathlib import Path

# Resolve database path relative to this script to avoid working-directory issues
BASE_DIR = Path(__file__).resolve().parent
db_path = BASE_DIR / 'sqllite' / 'debit_card_specializing.sqlite'
# Convert to string for sqlite3
db_path = str(db_path)

# Ensure conn exists if connection fails (so finally won't raise NameError)
conn = None

try:
    print("Start .")
    # 1. Connect to the database file
    conn = sqlite3.connect(db_path)
    print("Successfully connected to the database.")
    # Set the row_factory to sqlite3.Row to access columns by name
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # 2. Get the list of all tables in the database (exclude sqlite internal tables)
    sqlResult = cursor.execute("SELECT COUNT(GasStationID) FROM gasstations WHERE Country = 'CZE' AND Segment = 'Premium'")
    # Safely handle None and convert to string
    row = sqlResult.fetchone()
    if row:
        result_text = str(row[0])
        print("The ratio is: " + result_text)
    else:
        print("No data found.")

    '''tables = [row['name'] for row in cursor.fetchall()]
    db_data = {}
    for table_name in tables:
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 5;")
        # Convert each row to a dictionary
        rows = [dict(row) for row in cursor.fetchall()]
        db_data[table_name] = rows
    # 3. Write the dictionary to a JSON file
    output_json = BASE_DIR / "output_json_5.json"
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(db_data, f, indent=4)'''

    conn.close()
    #print(f"Successfully exported all tables to {output_json}")
except sqlite3.Error as e:
    print(f"Error connecting to database: {e}")

finally:
    # 4. Always close the connection
    if conn:
        conn.close()