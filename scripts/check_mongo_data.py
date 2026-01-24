import pymongo

def check_mongo():
    uri = "mongodb://localhost:27018"
    print(f"Connecting to {uri}...")
    try:
        client = pymongo.MongoClient(uri, serverSelectionTimeoutMS=2000)
        dbs = client.list_database_names()
        print("Databases found:", dbs)
        
        for db_name in dbs:
            if db_name in ['admin', 'config', 'local']:
                continue
            db = client[db_name]
            cols = db.list_collection_names()
            print(f"\nDB: {db_name}")
            print(f"Collections: {cols}")
            for col in cols:
                count = db[col].count_documents({})
                print(f"  - {col}: {count} docs")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_mongo()
