import json
import mongomock

def setup_mock_db(file_path):
    # 1. Initialize the mock client and database
    client = mongomock.MongoClient()
    db = client['debit_card_specializing_db']

    # 2. Load the JSON data
    data = json.load(file_path)

    # 3. Iterate through the JSON keys and create collections
    # Keys: CustomersCollection, ProductsCollection, etc.
    for collection_name, documents in data.items():
        if documents:  # Ensure there is data to insert
            db[collection_name].insert_many(documents)
            print(f"Inserted {len(documents)} documents into '{collection_name}'")

    return db
