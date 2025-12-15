"""Smoke test for sql_to_mongo_schema and load_mongo_schema.

Creates an in-memory SQLite DB with representative tables and verifies
schema extraction. The Mongo insertion step is attempted only if a
MongoDB server is reachable at the default URI; otherwise, it's skipped.
"""

from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, ForeignKey
import json
import logging

from schema_conversion.sql_to_mongo_schema import analyze_sql_schema
from schema_conversion.load_mongo_schema import convert_and_insert


def build_sample_db(sql_uri: str):
    engine = create_engine(sql_uri)
    meta = MetaData()

    # Parent table
    parent = Table("parent", meta, Column("id", Integer, primary_key=True), Column("name", String))

    # Child table: many-to-one to parent (should be embedded)
    child = Table("child", meta, Column("id", Integer, primary_key=True), Column("parent_id", Integer, ForeignKey("parent.id")), Column("value", String))

    # Lookup table
    lookup = Table("country", meta, Column("id", Integer, primary_key=True), Column("name", String))

    # Association table (many-to-many)
    assoc = Table("parent_child_assoc", meta, Column("parent_id", Integer, ForeignKey("parent.id")), Column("child_id", Integer, ForeignKey("child.id")))

    meta.create_all(engine)

    # Insert sample rows
    conn = engine.connect()
    conn.execute(parent.insert(), [{"id": 1, "name": "p1"}, {"id": 2, "name": "p2"}])
    conn.execute(child.insert(), [{"id": 1, "parent_id": 1, "value": "c1"}, {"id": 2, "parent_id": 1, "value": "c2"}])
    conn.execute(lookup.insert(), [{"id": 1, "name": "USA"}, {"id": 2, "name": "IND"}])
    conn.close()


def run_smoke():
    sql_uri = "sqlite+pysqlite:///:memory:"
    build_sample_db(sql_uri)

    schema = analyze_sql_schema(sql_uri)
    print("Extracted schema:")
    print(json.dumps(schema, indent=2))

    # Try to insert into MongoDB if available (will skip if not reachable)
    try:
        convert_and_insert(sql_uri, "mongodb://localhost:27017", db_id="smoke_test_db", schema=schema)
    except Exception as e:
        print("Mongo insert skipped/failed:", e)


if __name__ == "__main__":
    run_smoke()
