"""Smoke test for query generator inference."""

from models.query_generator.infer import QueryGenerator


def run_smoke():
    pred_schema = {
        "orders": ["id", "user_id", "amount", "status"],
        "users": ["id", "name", "country"]
    }
    nlqs = [
        "How many orders are there for users in USA?",
        "Total order amount per user",
        "Show orders where status = 'shipped'"
    ]

    gen = QueryGenerator()  # heuristic fallback
    for nlq in nlqs:
        pipeline = gen.predict(nlq, pred_schema)
        print('NLQ:', nlq)
        print('Pipeline:', pipeline)
        print('---')


if __name__ == '__main__':
    run_smoke()
