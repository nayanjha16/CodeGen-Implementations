from transformers import PreTrainedModel, PreTrainedTokenizer
import logging
from src.sql_parser import parse_sql
from src.mongo_converter import convert_to_mongo

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_sql(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    query: str, 
    schema_context: str
) -> str:
    """
    Generates SQL from a natural language query and schema context.
    
    Args:
        model: The pre-trained language model
        tokenizer: The tokenizer for the model
        query: Natural language query from user
        schema_context: Database schema information
        
    Returns:
        Generated SQL query as a string
    """
    prompt = f"""### Instruction:
You are an Expert SQL Developer. Given the database schema and a natural language question, generate a valid, optimized SQL query (SQLite compatible).

### Guidelines:
1. **JOINs**: Always use explicit `JOIN` clauses (e.g. `INNER JOIN`, `LEFT JOIN`) instead of comma-separated tables.
2. **Aggregations**: Use correct grouping. If using `COUNT`, `SUM`, `AVG`, ensure non-aggregated fields are in `GROUP BY`.
3. **Filtering**: Use `WHERE` for pre-aggregation filtering and `HAVING` for post-aggregation filtering (e.g. `HAVING COUNT(*) > 5`).
4. **Subqueries**: Use subqueries only when necessary (e.g. `WHERE id IN (SELECT ...)`).
5. **Output**: Return ONLY the SQL query. No specific explanations.

### Schema:
{schema_context}

### Question:
{query}

### Response:
"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
        
    # Clean up response
    # 1. Remove markdown code blocks
    if "```sql" in response:
        response = response.split("```sql")[1].split("```")[0].strip()
    elif "```" in response:
        response = response.split("```")[1].split("```")[0].strip()
        
    # 2. Remove any leading/trailing whitespace or quotes
    response = response.strip().strip('"').strip("'")
    
    # 3. Keep only the first statement if multiple are generated (simple heuristic)
    if ";" in response:
        response = response.split(";")[0] + ";"
        
    return response


import sqlite3
from src.sql_parser import parse_sql, is_complex, extract_complex_query_plan
from src.mongo_converter import convert_to_mongo, convert_complex_query_plan

def generate_nosql(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    sql_query: str,
    schema_context: str,
    db_path: str = None
) -> str:
    """
    Generates NoSQL (MongoDB) query from SQL query using a deterministic rule-based approach.
    Supports 'Complex Mode' which can execute subqueries against a real DB if db_path is provided.
    """
    try:
        logger.info(f"Converting SQL to NoSQL: {sql_query}")
        
        # 1. Parse SQL
        query_plan = extract_complex_query_plan(sql_query)
        logger.info(f"Query Plan Keys: {query_plan.keys()}")
        
        # 2. Check Complexity
        if is_complex(query_plan):
            logger.info("Complex Query Logic Detected (JOIN/HAVING/Subquery)")
            
            subquery_ids = {}
            # Execute subqueries if present and DB is available
            subqueries = query_plan.get("subqueries", [])
            if subqueries and db_path:
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    for sub in subqueries:
                         logger.info(f"Executing Subquery: {sub['sql']}")
                         cursor.execute(sub["sql"])
                         # Flatten results [ (1,), (2,) ] -> [1, 2]
                         ids = [row[0] for row in cursor.fetchall()]
                         subquery_ids[sub["field"]] = ids
                         logger.info(f"Subquery Result ({sub['field']}): {ids}")
                    conn.close()
                except Exception as e:
                    logger.error(f"Subquery Execution Failed: {e}")
                    return f"Error executing subquery: {e}"
            
            # Convert using Complex Converter
            mongo_query = convert_complex_query_plan(query_plan, subquery_ids)
            
        else:
            # Simple Mode
            mongo_query = convert_to_mongo(query_plan)
        
        logger.info(f"Generated MongoDB: {mongo_query}")
        return mongo_query
        
    except Exception as e:
        logger.error(f"Error converting SQL to NoSQL: {e}")
        return f"Error: {str(e)}"
