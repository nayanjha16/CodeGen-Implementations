
import os
import json
import sqlite3
try:
    import pymongo
    from pymongo import MongoClient
except ImportError:
    print("pymongo not installed, using mongomock")
    import mongomock
    from mongomock import MongoClient

def get_mongo_client(connection_string=None):
    if connection_string:
        return MongoClient(connection_string)
    return MongoClient() # Connects to localhost:27017 by default or mock

def setup_nosql_db(db_name="bird_nosql"):
    client = get_mongo_client()
    db = client[db_name]
    print(f"Connected to MongoDB: {db.name}")
    return db

if __name__ == "__main__":
    setup_nosql_db()
