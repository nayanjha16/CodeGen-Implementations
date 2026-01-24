import os
import json
import httpx
from sqlalchemy import create_engine, text
from pymongo import MongoClient
import pandas as pd
import datetime
from typing import List, Dict, Tuple

# We will need to import these properly once the package structure is final
# For now assuming they are in the python path or relative imports work if run as module
from app.schemas import QueryResponse
from services.inference.rag_service import RAGService
from services.inference.prompts import NL_TO_SQL_SYSTEM_PROMPT, SQL_TO_MQL_SYSTEM_PROMPT, REFLEXION_PROMPT
from services.migration.schema_discovery import SchemaDiscovery

INFERENCE_URL = os.getenv("INFERENCE_URL", "http://localhost:8000/v1")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")

class Orchestrator:
    def __init__(self):
        self.rag = RAGService()
        self.mongo_client = MongoClient(MONGO_URI)
        self.http_client = httpx.AsyncClient(timeout=300.0)
        if os.path.exists("/app/data"):
             self.data_path = "/app/data"
        else:
             # Fallback for local dev where CWD might be backend/
             self.data_path = os.path.abspath(os.path.join(os.getcwd(), "../data"))

    async def generate(self, db_id: str, question: str) -> QueryResponse:
        # 1. Schema Linking (RAG)
        # Verify db_path exists first (logic below is fine)
        # Correct path based on verification
        db_path = f"../data/minidev/MINIDEV/dev_databases/{db_id}/{db_id}.sqlite"
        if not os.path.exists(db_path):
             db_path = os.path.abspath(os.path.join(self.data_path, f"minidev/MINIDEV/dev_databases/{db_id}/{db_id}.sqlite"))

        schema_discovery = SchemaDiscovery(db_path)
        schema = schema_discovery.get_schema()

        # RAG Integration
        # Ingest schema into RAG service (in-memory for now)
        self.rag.ingest_schema(db_id, schema)
        
        # Retrieve top relevant tables (Reduced to 3 to fit context)
        relevant_table_names = self.rag.retrieve_relevant_tables(db_id, question, top_k=3)
        
        # Filter schema to only relevant tables
        filtered_schema = {k: v for k, v in schema.items() if k in relevant_table_names}
        
        # Fallback if RAG returns nothing
        if not filtered_schema:
            filtered_schema = schema

        # Simplify schema to save tokens (remove types, keys)
        # { table: [col1, col2] }
        simplified_schema = {
            table: [col['name'] for col in columns]
            for table, columns in filtered_schema.items()
        }
        schema_context = json.dumps(simplified_schema, indent=2)

        # 2. NL -> SQL
        try:
            sql_query = await self._query_llm(
                system_prompt=NL_TO_SQL_SYSTEM_PROMPT.format(schema_context=schema_context, question=question),
                user_prompt=question,
                model="Qwen/Qwen2.5-Coder-3B-Instruct-GPTQ-Int4" # Use base model for SQL generation
            )
        except Exception as e:
            print(f"LLM Call Failed: {e}")
            if hasattr(e, 'response'):
                 print(f"Response: {e.response.text}")
            return QueryResponse(
                sql_query="",
                mongo_pipeline=[],
                sql_result=[],
                mongo_result=[],
                execution_match=False,
                error=f"LLM Call Failed: {e}"
            )
        
        # Clean SQL (remove markdown)
        sql_query = sql_query.replace("```sql", "").replace("```", "").strip()

        # 3. Execute SQL (Reflexion Loop)
        sql_result, final_sql, error = self._execute_sql_with_retry(db_path, sql_query, schema_context)

        if error:
            return QueryResponse(
                sql_query=final_sql,
                mongo_pipeline=[],
                sql_result=[],
                mongo_result=[],
                execution_match=False,
                error=f"SQL Execution Failed: {error}"
            )

        # 4. SQL -> MQL
        # We need the Mongo Schema (which is effectively the SQL schema but in JSON structure)
        mql_prompt = SQL_TO_MQL_SYSTEM_PROMPT.format(
            sql_query=final_sql,
            mongo_schema=json.dumps(schema, indent=2) # Reuse schema for now
        )
        
        mql_str = await self._query_llm(
            system_prompt=mql_prompt,
            user_prompt="Convert to MongoDB Aggregation Pipeline",
            model="mql-adapter" # Use adapter for MQL conversion
        )
        
        # Clean MQL
        mql_str = mql_str.replace("```json", "").replace("```", "").strip()
        
        # Parse MQL from LLM output
        try:
            mongo_pipeline = json.loads(mql_str)
        except json.JSONDecodeError as e:
            print(f"ERROR: LLM returned invalid MQL: {mql_str[:200]}")
            print(f"JSON Error: {e}")
            return QueryResponse(
                sql_query=final_sql,
                mongo_pipeline=[],
                sql_result=sql_result,
                mongo_result=[],
                execution_match=False,
                error=f"Failed to parse generated MQL JSON: {str(e)}"
            )

        # 5. Execute MQL
        mongo_db = self.mongo_client[db_id]
        # Determine the collection. The pipeline usually starts with a collection, or we need to infer it.
        # Wait, the pipeline itself is [ {stage}, ... ]. It runs ON a collection.
        # But SQL joins multiple tables. Mongo aggregation must start on ONE collection ($lookup others).
        # We need to ask the LLM *which* collection to start on, OR allow the LLM to output { "collection": "...", "pipeline": [...] }
        # Let's simple heuristic: look at the first FROM table in SQL (not robust) or ask LLM.
        # Update Prompt strategy might be needed later.
        # For now, let's assume the LLM output includes the collection or we guess from SQL.
        
        # TEMP FIX: Extract table from SQL 'FROM' clause simple parse
        start_collection = self._extract_start_table(final_sql)
        
        mongo_result = []
        try:
            cursor = mongo_db[start_collection].aggregate(mongo_pipeline)
            mongo_result = list(cursor)
            # Convert ObjectIds and Dates to strings for comparison
            mongo_result = self._serialize_mongo(mongo_result)
        except Exception as e:
             return QueryResponse(
                sql_query=final_sql,
                mongo_pipeline=mongo_pipeline,
                sql_result=sql_result,
                mongo_result=[],
                execution_match=False,
                error=f"Mongo Execution Failed: {e}"
            )

        # 6. Compare
        match = self._compare_results(sql_result, mongo_result)

        return QueryResponse(
            sql_query=final_sql,
            mongo_pipeline=mongo_pipeline,
            sql_result=sql_result,
            mongo_result=mongo_result,
            execution_match=match
        )

    async def _query_llm(self, system_prompt: str, user_prompt: str, model: str = "Qwen/Qwen2.5-Coder-3B-Instruct-GPTQ-Int4") -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "max_tokens": 1024
        }
        try:
            response = await self.http_client.post(f"{INFERENCE_URL}/chat/completions", json=payload)
            response.raise_for_status()
            return response.json()['choices'][0]['message']['content']
        except Exception as e:
            # Fallback for dev without vLLM
            print(f"LLM Call Failed: {e}")
            return "SELECT * FROM error"

    def _execute_sql_with_retry(self, db_path, sql, schema_context):
        engine = create_engine(f"sqlite:///{db_path}")
        try:
            with engine.connect() as conn:
                result = conn.execute(text(sql))
                return [dict(row) for row in result.mappings()], sql, None
        except Exception as e:
            # Reflexion would go here
            # For this pass, just return error
            return [], sql, str(e)

    def _extract_start_table(self, sql):
        # Very naive parser
        lower_sql = sql.lower()
        if "from" in lower_sql:
            parts = lower_sql.split("from")
            if len(parts) > 1:
                table_part = parts[1].strip().split(" ")[0]
                return table_part.strip(";,")
        return "unknown_table"

    def _serialize_mongo(self, data):
        # Helper to make Mongo results JSON serializable and comparable
        new_data = []
        for doc in data:
            new_doc = {}
            for k, v in doc.items():
                if k == '_id': continue # Skip ID for comparison usually
                if isinstance(v, (datetime.datetime, datetime.date)):
                    new_doc[k] = v.isoformat()
                else:
                    new_doc[k] = v
            new_data.append(new_doc)
        return new_data

    def _compare_results(self, sql_res, mongo_res):
        """
        Compare two lists of dictionaries (records).
        Strategy:
        1. Normalize types (dates to strings, floats to fixed precision).
        2. Convert list of dicts to set of frozensets (for order-independent comparison).
        """
        if len(sql_res) != len(mongo_res):
            return False
            
        def normalize_item(item):
            # Convert dict to sorted tuple of items for hashing
            # Handle float tolerance
            norm = []
            keys = sorted(item.keys())
            
            # Special logic: If there is only one key, ignore the key name for comparison
            # This handles cases like {"count": 10} vs {"COUNT(*)": 10}
            if len(keys) == 1:
                val = item[keys[0]]
                if isinstance(val, float):
                    val = round(val, 2)
                norm.append(('__single_value__', str(val)))
            else:
                for k in keys:
                    val = item[k]
                    # Normalize types
                    if isinstance(val, float):
                        if pd.isna(val): # Handle NaN
                            val = None
                        else:
                            val = round(val, 2)
                    
                    # Convert to string
                    val_str = str(val)
                    
                    # Normalize various representations of None/NaN
                    if val_str in ("nan", "None", ""):
                        val_str = "None"
                    
                    # Normalize numeric strings (e.g., "0.0" -> "0", "1.0" -> "1")
                    try:
                        float_val = float(val_str)
                        if float_val.is_integer():
                            val_str = str(int(float_val))
                        else:
                            val_str = str(round(float_val, 2))
                    except (ValueError, AttributeError):
                        pass  # Not a number, keep as-is
                    
                    norm.append((k, val_str))
            return frozenset(norm)

        sql_set = {normalize_item(i) for i in sql_res}
        mongo_set = {normalize_item(i) for i in mongo_res}
        
        
        return sql_set == mongo_set
