"""
Dataset formatter for Qwen instruction tuning.
Converts Spider dataset examples into the chat format expected by Qwen.
"""
from typing import List, Dict, Any
import json

def format_spider_example(example: Any, schema: str) -> List[Dict[str, str]]:
    """
    Formats a single Spider example into Qwen chat format.
    
    Args:
        example: SpiderExample object or dict with 'question' and 'query'
        schema: Database schema string
        
    Returns:
        List of message dicts [{"role": "user", ...}, {"role": "assistant", ...}]
    """
    question = example.question if hasattr(example, 'question') else example['question']
    sql = example.query if hasattr(example, 'query') else example['query']
    
    # System prompt is implicit in Qwen, but we can add it as user context if needed.
    # Standard Qwen format:
    # <|im_start|>user
    # ... <|im_end|>
    # <|im_start|>assistant
    # ... <|im_end|>
    
    user_content = f"""You are a SQL query generator. Given a database schema and question, output ONLY the SQL query.

DATABASE SCHEMA:
{schema}

QUESTION: {question}

SQL:"""

    return [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": sql}
    ]

def prepare_training_data(examples: List[Any], schema_loader_fn) -> List[Dict[str, Any]]:
    """
    Prepares a list of examples for training.
    
    Args:
        examples: List of SpiderExample objects
        schema_loader_fn: Function to get schema for a db_id
        
    Returns:
        List of dicts with 'messages' key
    """
    formatted_data = []
    
    for ex in examples:
        try:
            db_id = ex.db_id if hasattr(ex, 'db_id') else ex['db_id']
            schema = schema_loader_fn(db_id)
            messages = format_spider_example(ex, schema)
            formatted_data.append({"messages": messages})
        except Exception as e:
            # Skip examples where schema loading fails
            continue
            
    return formatted_data
