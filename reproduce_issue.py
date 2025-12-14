
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from dataset_loader import get_database_schema, load_spider_tables
from inference_improved import correct_sql_columns

def test_schema_keys():
    print("Testing Schema Key Inclusion...")
    # Find a database with known keys
    # Arbitrarily pick one that is likely to have relationships
    # 'concert_singer' is one used in the examples
    db_id = 'concert_singer'
    
    try:
        schema = get_database_schema(db_id)
        print(f"Schema for {db_id} (first 500 chars):\n{schema[:500]}...")
        
        if "PRIMARY KEY" in schema:
            print("[PASS] PRIMARY KEY found in schema.")
        else:
            print("[FAIL] PRIMARY KEY not found in schema.")
            
        if "FOREIGN KEY" in schema:
            print("[PASS] FOREIGN KEY found in schema.")
        else:
            print("[FAIL] FOREIGN KEY not found in schema.")
            
        return schema
    except Exception as e:
        print(f"[ERROR] Failed to load schema: {e}")
        return ""

def test_fuzzy_correction(schema):
    print("\nTesting Fuzzy Column Correction...")
    
    # Test case from user request: petage -> pet_age
    # We need a schema that has 'pet_age'. 'pets_1' often has this.
    
    # Let's mock a schema string for control
    mock_schema = """
    CREATE TABLE pets (
      petid INTEGER,
      pet_age INTEGER,
      pet_name TEXT
    )
    """
    
    bad_sql = "SELECT petage FROM pets"
    corrected_sql = correct_sql_columns(bad_sql, mock_schema)
    
    print(f"Original: {bad_sql}")
    print(f"Corrected: {corrected_sql}")
    
    if "pet_age" in corrected_sql:
        print("[PASS] 'petage' corrected to 'pet_age'")
    else:
        print(f"[FAIL] Correction failed. Expected 'pet_age', got '{corrected_sql}'")

    # Double check avoiding over-correction
    bad_sql_2 = "SELECT petid FROM pets" # valid
    corrected_sql_2 = correct_sql_columns(bad_sql_2, mock_schema)
    if corrected_sql_2 == bad_sql_2:
        print("[PASS] Valid column 'petid' preserved")
    else:
        print(f"[FAIL] 'petid' was altered to '{corrected_sql_2}'")

if __name__ == "__main__":
    schema = test_schema_keys()
    test_fuzzy_correction(schema)
