"""Smoke test for schema predictor inference.

Uses heuristic fallback (no model) to demonstrate that inference returns valid JSON
and only fields present in schema.
"""

from models.schema_predictor.infer import SchemaPredictor


def run_smoke():
    schema = {
        "users": ["id", "name", "age", "country"],
        "orders": ["id", "user_id", "amount", "created_at"],
    }

    nlq = "Find users and their total order amounts where country is USA"

    pred = SchemaPredictor()  # no model => heuristic fallback
    out = pred.predict(nlq, schema)
    print(out)


if __name__ == "__main__":
    run_smoke()
