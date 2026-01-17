import json
import re

from llm.llm_client import call_llm
from rag.sql_analyzer import analyze_sql
from rag.schema_store import load_schema
from rag.retriever import retrieve_rag_context


def extract_json_array(text: str):
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        raise ValueError("No JSON array found in LLM output")
    return json.loads(match.group())


def sql_to_mongo_pipeline(sql, sqlite_db_path, prompt_template_path):
    # -------------------------
    # RAG v3
    # -------------------------
    sql_info = analyze_sql(sql)
    schema_graph = load_schema(sqlite_db_path)
    rag_context = retrieve_rag_context(sql_info, schema_graph)

    # -------------------------
    # PROMPT
    # -------------------------
    with open(prompt_template_path) as f:
        template = f.read()

    prompt = template.format(
        sql=sql,
        root_collection=rag_context["root_collection"],
        join_templates=json.dumps(rag_context["join_templates"], indent=2)
    )

    # -------------------------
    # LLM CALL
    # -------------------------
    response = call_llm(prompt)

    mongo_pipeline = extract_json_array(response)

    return mongo_pipeline, rag_context["root_collection"]
