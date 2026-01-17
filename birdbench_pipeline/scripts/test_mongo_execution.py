from pipelines.mongo_executor import execute_mongo_query
from evaluation.result_normalizer import normalize_results

# --------------------------------------------------
# Manual test for debit_card_specializing
# --------------------------------------------------

mongo_db = "debit_card_specializing"
collection = "customers"

# Example Mongo pipeline:
# Count EUR and CZK customers
pipeline = [
    {
        "$group": {
            "_id": None,
            "eur_count": {
                "$sum": {
                    "$cond": [{"$eq": ["$Currency", "EUR"]}, 1, 0]
                }
            },
            "czk_count": {
                "$sum": {
                    "$cond": [{"$eq": ["$Currency", "CZK"]}, 1, 0]
                }
            }
        }
    },
    {
        "$project": {
            "_id": 0,
            "ratio": {
                "$cond": [
                    {"$eq": ["$czk_count", 0]},
                    None,
                    {"$divide": ["$eur_count", "$czk_count"]}
                ]
            }
        }
    }
]

# --------------------------------------------------
# Execute Mongo query
# --------------------------------------------------
mongo_results = execute_mongo_query(
    mongo_db_name=mongo_db,
    collection_name=collection,
    pipeline=pipeline
)

normalized = normalize_results(mongo_results)

print("\n=== MONGO QUERY RESULT ===")
for row in normalized:
    print(row)
