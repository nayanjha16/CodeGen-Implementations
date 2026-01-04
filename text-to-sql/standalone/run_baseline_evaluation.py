"""
================================================================================
TEXT-TO-SQL BASELINE EVALUATION - ALL-IN-ONE SCRIPT
================================================================================

This is a consolidated version of the baseline evaluation pipeline.
All code is in one file for easy understanding of the flow.

FLOW:
1. Configuration (line ~30)
2. Load Spider Dataset (line ~60)
3. Load Model (line ~180)
4. Generate SQL (line ~220)
5. Evaluate Results (line ~270)
6. Main Evaluation Loop (line ~330)

To run:
    uv run python standalone/run_baseline_evaluation.py --limit 10
    
================================================================================
"""

import os
import sys
import json
import re
import sqlite3
import argparse
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==============================================================================
# SECTION 1: CONFIGURATION
# ==============================================================================

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SPIDER_DIR = os.path.join(DATA_DIR, 'spider')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')

# Model Configuration
MODEL_NAME = "Qwen/Qwen2.5-Coder-0.5B-Instruct"  # 494M parameters, baseline model

# Hardware Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"""
================================================================================
CONFIGURATION
================================================================================
Model: {MODEL_NAME}
Device: {DEVICE}
Spider Dir: {SPIDER_DIR}
Output Dir: {OUTPUT_DIR}
================================================================================
""")


# ==============================================================================
# SECTION 2: DATASET LOADING
# ==============================================================================

class SpiderExample:
    """
    Represents a single example from the Spider dataset.
    
    Attributes:
        question: Natural language question (e.g., "How many singers are there?")
        query: Ground truth SQL query (e.g., "SELECT COUNT(*) FROM singer")
        db_id: Database identifier (e.g., "concert_singer")
        db_path: Path to SQLite database file
        complexity: "simple" or "complex" based on query structure
    """
    
    def __init__(self, question: str, query: str, db_id: str, db_path: Optional[str] = None):
        self.question = question
        self.query = query
        self.db_id = db_id
        self.db_path = db_path
        self.complexity = self._calculate_complexity()

    def _calculate_complexity(self) -> str:
        """Determine if query is simple or complex based on SQL features."""
        sql = self.query.lower()
        # Complex if: nested SELECT, JOIN, UNION, INTERSECT, EXCEPT
        if sql.count("select") > 1:
            return "complex"
        if any(kw in sql for kw in ["join", "union", "intersect", "except"]):
            return "complex"
        return "simple"
    
    def __repr__(self) -> str:
        return f"SpiderExample(db='{self.db_id}', complexity='{self.complexity}', q='{self.question[:40]}...')"


def load_spider_dataset(split: str = 'dev', limit: Optional[int] = None) -> List[SpiderExample]:
    """
    Load Spider dataset from JSON files.
    
    Args:
        split: 'train' or 'dev'
        limit: Maximum number of examples to load
        
    Returns:
        List of SpiderExample objects
    """
    # Determine file path
    json_filename = f'{split}.json'
    if split == 'train' and not os.path.exists(os.path.join(SPIDER_DIR, json_filename)):
        json_filename = 'train_spider.json'
        
    json_path = os.path.join(SPIDER_DIR, json_filename)
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(
            f"Dataset not found: {json_path}\n"
            f"Run: uv run python scripts/download_spider.py"
        )
    
    print(f"Loading Spider {split} dataset from: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Parse examples
    examples = []
    database_dir = os.path.join(SPIDER_DIR, 'database')
    
    for item in data:
        db_id = item['db_id']
        db_path = os.path.join(database_dir, db_id, f'{db_id}.sqlite')
        
        example = SpiderExample(
            question=item['question'],
            query=item['query'],
            db_id=db_id,
            db_path=db_path if os.path.exists(db_path) else None
        )
        examples.append(example)
        
        if limit and len(examples) >= limit:
            break
    
    print(f"Loaded {len(examples)} examples from Spider {split} set")
    return examples


def load_spider_tables() -> Dict[str, Any]:
    """Load database schema information from tables.json."""
    tables_path = os.path.join(SPIDER_DIR, 'tables.json')
    
    with open(tables_path, 'r', encoding='utf-8') as f:
        tables = json.load(f)
    
    return {table['db_id']: table for table in tables}


def get_database_schema(db_id: str) -> str:
    """
    Generate CREATE TABLE statements for a database.
    
    This provides the schema context that the model needs to generate SQL.
    Includes PRIMARY KEY and FOREIGN KEY constraints for JOIN hints.
    """
    tables = load_spider_tables()
    
    if db_id not in tables:
        raise KeyError(f"Database '{db_id}' not found")
    
    db_info = tables[db_id]
    schema_lines = []
    
    table_names = db_info['table_names_original']
    column_names = db_info['column_names_original']
    column_types = db_info['column_types']
    
    # Group columns by table
    tables_columns = {}
    for col_idx, (table_idx, col_name) in enumerate(column_names):
        if table_idx == -1:  # Skip the * column
            continue
        
        table_name = table_names[table_idx]
        if table_name not in tables_columns:
            tables_columns[table_name] = []
        
        col_type = column_types[col_idx]
        tables_columns[table_name].append((col_idx, col_name, col_type))
    
    # Generate CREATE TABLE statements
    for table_idx, table_name in enumerate(table_names):
        if table_name not in tables_columns:
            continue
            
        schema_lines.append(f"CREATE TABLE {table_name} (")
        
        # Add columns
        for col_idx, col_name, col_type in tables_columns[table_name]:
            schema_lines.append(f"  {col_name} {col_type},")
        
        # Add Primary Keys
        primary_keys = db_info.get('primary_keys', [])
        pk_columns = []
        for pk_idx in primary_keys:
            if pk_idx < len(column_names):
                pk_table_idx, pk_col_name = column_names[pk_idx]
                if pk_table_idx == table_idx:
                    pk_columns.append(pk_col_name)
        
        if pk_columns:
            schema_lines.append(f"  PRIMARY KEY ({', '.join(pk_columns)}),")
        
        # Add Foreign Keys
        foreign_keys = db_info.get('foreign_keys', [])
        for source_idx, target_idx in foreign_keys:
            if source_idx < len(column_names) and target_idx < len(column_names):
                source_table_idx, source_col = column_names[source_idx]
                target_table_idx, target_col = column_names[target_idx]
                
                if source_table_idx == table_idx:
                    target_table_name = table_names[target_table_idx]
                    schema_lines.append(f"  FOREIGN KEY ({source_col}) REFERENCES {target_table_name}({target_col}),")
        
        # Remove trailing comma and close
        if schema_lines[-1].endswith(','):
            schema_lines[-1] = schema_lines[-1][:-1]
        schema_lines.append(")")
        schema_lines.append("")
    
    return "\n".join(schema_lines)


# ==============================================================================
# SECTION 3: MODEL LOADING
# ==============================================================================

def load_model() -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the Qwen2.5-Coder model and tokenizer.
    
    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {MODEL_NAME}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Load model
    print(f"Loading model on {DEVICE}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True
    )
    
    if DEVICE == "cpu":
        model = model.to("cpu")
    
    print(f"Model loaded successfully on {DEVICE}")
    return model, tokenizer


# ==============================================================================
# SECTION 4: SQL GENERATION (INFERENCE)
# ==============================================================================

def generate_sql(model, tokenizer, query: str, schema_context: str) -> str:
    """
    Generate SQL from a natural language query.
    
    This is the core inference function that:
    1. Constructs a structured prompt
    2. Tokenizes the input
    3. Generates output tokens
    4. Decodes and cleans the response
    
    Args:
        model: The loaded language model
        tokenizer: The tokenizer
        query: Natural language question
        schema_context: Database schema (CREATE TABLE statements)
        
    Returns:
        Generated SQL query string
    """
    
    # Step 1: Construct structured prompt
    prompt = f"""### Instruction:
You are a text-to-SQL generator. Given the database schema and a natural language question, generate a valid SQL query.
Return ONLY the SQL query, without any explanation or markdown formatting.

### Schema:
{schema_context}

### Question:
{query}

### Response:
"""

    # Step 2: Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Step 3: Generate output tokens
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,      # Maximum tokens to generate
        do_sample=False,         # Greedy decoding (deterministic)
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Step 4: Decode tokens to text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Step 5: Extract SQL from response
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    # Step 6: Clean up response
    # Remove markdown code blocks
    if "```sql" in response:
        response = response.split("```sql")[1].split("```")[0].strip()
    elif "```" in response:
        response = response.split("```")[1].split("```")[0].strip()
    
    # Remove quotes
    response = response.strip().strip('"').strip("'")
    
    # Keep only first statement
    if ";" in response:
        response = response.split(";")[0] + ";"
    
    return response


# ==============================================================================
# SECTION 5: EVALUATION METRICS
# ==============================================================================

def normalize_sql(sql: str) -> str:
    """
    Normalize SQL for comparison.
    
    Steps:
    1. Convert to lowercase
    2. Normalize whitespace
    3. Remove trailing semicolon
    """
    sql = sql.lower()
    sql = re.sub(r'\s+', ' ', sql)
    sql = sql.rstrip(';').strip()
    return sql


def compute_exact_match(predicted: str, reference: str) -> bool:
    """
    Check if predicted SQL exactly matches reference (after normalization).
    
    This is a strict metric - even semantically equivalent queries may fail.
    """
    return normalize_sql(predicted) == normalize_sql(reference)


def execute_sql(sql: str, db_path: str) -> Tuple[bool, Optional[list], Optional[str]]:
    """
    Execute SQL query on SQLite database.
    
    Returns:
        Tuple of (success, results, error_message)
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        conn.close()
        return True, results, None
    except Exception as e:
        return False, None, str(e)


def compute_execution_accuracy(predicted: str, reference: str, db_path: str) -> bool:
    """
    Check if predicted and reference SQL return the same results.
    
    This is a more meaningful metric than exact match because:
    - Multiple SQL syntaxes can produce the same result
    - Measures actual correctness, not just string matching
    """
    # Execute predicted query
    pred_success, pred_results, pred_error = execute_sql(predicted, db_path)
    if not pred_success:
        return False
    
    # Execute reference query
    ref_success, ref_results, ref_error = execute_sql(reference, db_path)
    if not ref_success:
        return False
    
    # Compare results (order-insensitive)
    try:
        pred_set = set(tuple(row) for row in pred_results)
        ref_set = set(tuple(row) for row in ref_results)
        return pred_set == ref_set
    except TypeError:
        return sorted(pred_results) == sorted(ref_results)


# ==============================================================================
# SECTION 6: MAIN EVALUATION LOOP
# ==============================================================================

def main():
    """
    Main evaluation function.
    
    Flow:
    1. Parse command line arguments
    2. Load the model
    3. Load the dataset
    4. For each example:
       a. Get database schema
       b. Generate SQL using model
       c. Compute exact match
       d. Compute execution accuracy
    5. Save results to JSON
    """
    
    # Parse arguments
    parser = argparse.ArgumentParser(description="Baseline Text-to-SQL Evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--complex-only", action="store_true", help="Evaluate only complex queries")
    args = parser.parse_args()
    
    print("=" * 60)
    print("BASELINE TEXT-TO-SQL EVALUATION")
    print("=" * 60)
    
    # Step 1: Load model
    model, tokenizer = load_model()
    
    # Step 2: Load dataset
    examples = load_spider_dataset(split='dev', limit=args.limit if not args.complex_only else None)
    
    if args.complex_only:
        examples = [e for e in examples if e.complexity == 'complex']
        if args.limit:
            examples = examples[:args.limit]
    
    print(f"\nEvaluating {len(examples)} examples...\n")
    
    # Step 3: Evaluate each example
    results = []
    exact_matches = 0
    execution_matches = 0
    
    for example in tqdm(examples, desc="Evaluating"):
        # Get schema for this database
        schema = get_database_schema(example.db_id)
        
        # Generate SQL
        try:
            generated_sql = generate_sql(model, tokenizer, example.question, schema)
        except Exception as e:
            print(f"Error generating SQL: {e}")
            generated_sql = "SELECT * FROM error"
        
        # Compute metrics
        exact_match = compute_exact_match(generated_sql, example.query)
        execution_match = False
        
        if example.db_path and os.path.exists(example.db_path):
            execution_match = compute_execution_accuracy(
                generated_sql, example.query, example.db_path
            )
        
        # Track results
        if exact_match:
            exact_matches += 1
        if execution_match:
            execution_matches += 1
        
        results.append({
            "question": example.question,
            "gold_query": example.query,
            "generated_query": generated_sql,
            "db_id": example.db_id,
            "exact_match": exact_match,
            "execution_match": execution_match,
            "complexity": example.complexity
        })
    
    # Step 4: Print summary
    total = len(results)
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Examples: {total}")
    print(f"Exact Match Accuracy: {exact_matches}/{total} ({100*exact_matches/total:.2f}%)")
    print(f"Execution Accuracy: {execution_matches}/{total} ({100*execution_matches/total:.2f}%)")
    print("=" * 60)
    
    # Step 5: Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, 'baseline_results.json')
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

