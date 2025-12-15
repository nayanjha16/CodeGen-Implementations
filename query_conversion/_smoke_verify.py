"""Smoke test for SQL->Mongo conversion and verification.

Builds a small in-memory SQLite DB, runs sample SQL queries, converts them
to pipelines, and verifies results using the in-memory pipeline runner.
"""

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, ForeignKey
from query_conversion.sql_to_mongo_query import sql_to_mongo_pipeline
from query_conversion.verify_execution import run_sql, run_pipeline_on_docs, compare_results


def build_db(sql_uri: str):
    engine = create_engine(sql_uri)
    meta = MetaData()
    users = Table('users', meta, Column('id', Integer, primary_key=True), Column('name', String), Column('age', Integer))
    orders = Table('orders', meta, Column('id', Integer, primary_key=True), Column('user_id', Integer, ForeignKey('users.id')), Column('amount', Integer))
    meta.create_all(engine)
    conn = engine.connect()
    conn.execute(users.insert(), [{'id': 1, 'name': 'Alice', 'age': 30}, {'id': 2, 'name': 'Bob', 'age': 25}])
    conn.execute(orders.insert(), [{'id': 1, 'user_id': 1, 'amount': 10}, {'id': 2, 'user_id': 1, 'amount': 20}, {'id': 3, 'user_id': 2, 'amount': 5}])
    conn.close()


def run_smoke():
    sql_uri = 'sqlite+pysqlite:///:memory:'
    build_db(sql_uri)
    engine = create_engine(sql_uri)

    queries = [
        "SELECT user_id, SUM(amount) as total FROM orders GROUP BY user_id ORDER BY total DESC",
        "SELECT name FROM users WHERE age > 26",
        "SELECT u.name, COUNT(o.id) as cnt FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.name",
    ]

    for q in queries:
        conv = sql_to_mongo_pipeline(q)
        print('SQL:', q)
        print('Pipeline:', conv)
        sql_rows = run_sql(engine, q)
        mongo_rows = run_pipeline_on_docs(sql_rows, conv['pipeline']) if conv else []
        print('SQL rows:', sql_rows)
        print('Mongo rows:', mongo_rows)
        print('Match:', compare_results(sql_rows, mongo_rows))
        print('---')


if __name__ == '__main__':
    run_smoke()
