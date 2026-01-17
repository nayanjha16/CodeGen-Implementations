from typing import List, Dict, Any
from pymongo import MongoClient


def execute_mongo_query(
    mongo_db_name: str,
    collection_name: str,
    pipeline: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    Execute a MongoDB aggregation pipeline and return results
    as a list of dictionaries.
    """
    client = MongoClient("mongodb://localhost:27017/")
    db = client[mongo_db_name]
    collection = db[collection_name]

    cursor = collection.aggregate(pipeline)

    results = []
    for doc in cursor:
        doc.pop("_id", None)  # Remove MongoDB internal ID
        results.append(doc)

    return results
