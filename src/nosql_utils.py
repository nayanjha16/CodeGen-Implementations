"""
Utilities for NoSQL (MongoDB) operations.

This module provides functions to:
1. Initialize a mongomock client with data from a SQLite database.
2. Execute MongoDB queries (MQL) against the mock database.
3. Convert result sets for comparison.
"""

import sqlite3
import mongomock
import json
import re

def load_sqlite_to_mongo(db_path):
    """
    Load data from a SQLite database into a mongomock client.
    
    Args:
        db_path: Path to SQLite database
        
    Returns:
        mongomock.MongoClient: A client with the database populated
    """
    client = mongomock.MongoClient()
    db = client.db
    
    if not db_path:
        return client
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    
    for table in tables:
        if table.startswith('sqlite_'):
            continue
            
        cursor.execute(f"SELECT * FROM \"{table}\"")
        columns = [description[0] for description in cursor.description]
        rows = cursor.fetchall()
        
        documents = []
        for row in rows:
            doc = dict(zip(columns, row))
            # Convert values to simple types if needed
            documents.append(doc)
            
        if documents:
            db[table].insert_many(documents)
            
    conn.close()
    return client

def execute_mongo_query(client, query_str):
    """
    Execute a text-based MongoDB query string against a mongomock client.
    
    Supported formats:
    - db.collection.find({ ... })
    - db.collection.aggregate([ ... ])
    - db.collection.countDocuments({ ... })
    
    Args:
        client: mongomock.MongoClient
        query_str: The query string (e.g., "db.users.find({'age': 25})")
        
    Returns:
        list: Result documents
        str: Error message (if any)
    """
    try:
        db = client.db
        
        # Clean query
        query_str = query_str.strip()
        if not query_str.startswith('db.'):
            return None, "Query must start with 'db.'"
            
        # Extract collection and operation
        # Regex to capture: db.<collection>.<operation>(<args>)
        match = re.match(r"db\.(\w+)\.(\w+)\((.*)\)", query_str, re.DOTALL)
        
        if not match:
            return None, "Invalid query format"
            
        collection_name = match.group(1)
        operation = match.group(2)
        args_str = match.group(3)
        
        collection = db[collection_name]
        
        # Parse arguments (basic JSON parsing)
        # This is simplified; assumes args are valid JSON-like structure
        # We replace single quotes with double quotes for valid JSON parsing if needed
        # But MQL often uses single quotes.
        
        # Helper to safely parse basic args
        def parse_args(s):
            # Attempt to eval as python objects (dict/list)
            # DANGEROUS in prod, acceptable for local eval demo
            return eval(s) 
            
        args = parse_args(args_str)
        
        if operation == 'find':
            # find(filter, projection)
            filter_doc = args if isinstance(args, dict) else (args[0] if args else {})
            projection = args[1] if isinstance(args, tuple) and len(args) > 1 else None
            
            cursor = collection.find(filter_doc, projection)
            return list(cursor), None
            
        elif operation == 'aggregate':
            pipeline = args if isinstance(args, list) else []
            cursor = collection.aggregate(pipeline)
            return list(cursor), None
            
        elif operation == 'countDocuments' or operation == 'count':
            filter_doc = args if isinstance(args, dict) else {}
            count = collection.count_documents(filter_doc)
            return [{'count': count}], None
            
        else:
            return None, f"Unsupported operation: {operation}"
            
    except Exception as e:
        return None, str(e)

def compare_results(sql_results, mongo_results):
    """
    Compare execution results from SQL and NoSQL.
    
    This is complex because structure differs:
    - SQL: List of tuples (rows)
    - Mongo: List of dicts (documents)
    
    Strategy:
    1. Flatten Mongo results to values only, sorted.
    2. Flatten SQL results to values only, sorted.
    3. Compare sets/lists.
    """
    def flatten(data):
        flat = []
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    # Sort keys to ensure deterministic order of values
                    flat.extend([str(v) for k, v in sorted(item.items()) if k != '_id'])
                elif isinstance(item, (tuple, list)):
                    flat.extend([str(x) for x in item])
                else:
                    flat.append(str(item))
        return sorted(flat)
        
    flat_sql = flatten(sql_results)
    flat_mongo = flatten(mongo_results)
    
    return flat_sql == flat_mongo
