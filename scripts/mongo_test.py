
import pymongo
import sys

def verify_mongo():
    print("--- MongoDB Verification Script ---")
    try:
        # 1. Connect
        print("1. Connecting to MongoDB (localhost:27017)...")
        client = pymongo.MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
        client.server_info() # Trigger connection check
        print("   [SUCCESS] Connected.")
        
        # 2. Create Database
        print("2. Creating/Accessing 'test_database'...")
        db = client["test_database"]
        
        # 3. Create Collection & Insert
        print("3. Creating 'test_collection' and inserting data...")
        collection = db["test_collection"]
        sample_doc = {"name": "MongoDB Test", "status": "Active", "id": 1}
        result = collection.insert_one(sample_doc)
        print(f"   [SUCCESS] Inserted document ID: {result.inserted_id}")
        
        # 4. Find/Query
        print("4. Querying the document back...")
        found_doc = collection.find_one({"name": "MongoDB Test"})
        if found_doc:
            print(f"   [SUCCESS] Found document: {found_doc}")
        else:
            print("   [ERROR] Document not found!")
            
        # Cleanup
        print("5. Cleaning up (dropping test database)...")
        client.drop_database("test_database")
        print("   [SUCCESS] Test completed.")
        
    except pymongo.errors.ServerSelectionTimeoutError:
        print("\n[ERROR] Could not connect to MongoDB.")
        print("Is the service running? Try running 'net start MongoDB' as Admin.")
    except Exception as e:
        print(f"\n[ERROR] Unexpected error: {e}")

if __name__ == "__main__":
    verify_mongo()
