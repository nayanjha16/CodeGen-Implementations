"""
================================================================================
TEXT-TO-SQL IMPROVED EVALUATION - ALL-IN-ONE SCRIPT
================================================================================

This is a consolidated version of the IMPROVED evaluation pipeline.
All code is in one file for easy understanding of the flow.

KEY IMPROVEMENTS over baseline:
1. Uses larger model (1.5B instead of 0.5B)
2. Better stop conditions to prevent rambling
3. FUZZY COLUMN CORRECTION - fixes hallucinated column names

FLOW:
1. Configuration (line ~35)
2. Load Spider Dataset (line ~70)
3. Load Model (line ~200)
4. Generate SQL with Improvements (line ~240)
5. Fuzzy Column Correction (line ~300)
6. Evaluate Results (line ~400)
7. Main Evaluation Loop (line ~460)

To run:
    uv run python standalone/run_improved_evaluation.py --limit 10
    
================================================================================
"""

import os
import sys
import json
import re
import sqlite3
import argparse
import difflib  # For fuzzy matching
from typing import List, Dict, Any, Optional, Tuple
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel  # For loading LoRA adapter


# ==============================================================================
# SECTION 1: CONFIGURATION
# ==============================================================================

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
SPIDER_DIR = os.path.join(DATA_DIR, 'spider')
OUTPUT_DIR = os.path.join(BASE_DIR, 'results')
ADAPTER_PATH = os.path.join(BASE_DIR, 'results', 'checkpoints', 'final_checkpoint')

# Model Configuration - IMPROVED uses larger 1.5B model
MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"  # 1.5B parameters

# Hardware Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f"""
================================================================================
CONFIGURATION (IMPROVED)
================================================================================
Model: {MODEL_NAME}
Device: {DEVICE}
Spider Dir: {SPIDER_DIR}
Output Dir: {OUTPUT_DIR}
================================================================================
""")


# ==============================================================================
# SECTION 2: DATASET LOADING (Same as baseline)
# ==============================================================================

class SpiderExample:
    """Represents a single example from the Spider dataset."""
    
    def __init__(self, question: str, query: str, db_id: str, db_path: Optional[str] = None):
        self.question = question
        self.query = query
        self.db_id = db_id
        self.db_path = db_path
        self.complexity = self._calculate_complexity()

    def _calculate_complexity(self) -> str:
        sql = self.query.lower()
        if sql.count("select") > 1:
            return "complex"
        if any(kw in sql for kw in ["join", "union", "intersect", "except"]):
            return "complex"
        return "simple"
    
    def __repr__(self) -> str:
        return f"SpiderExample(db='{self.db_id}', complexity='{self.complexity}')"


def load_spider_dataset(split: str = 'dev', limit: Optional[int] = None) -> List[SpiderExample]:
    """Load Spider dataset from JSON files."""
    json_filename = f'{split}.json'
    if split == 'train' and not os.path.exists(os.path.join(SPIDER_DIR, json_filename)):
        json_filename = 'train_spider.json'
        
    json_path = os.path.join(SPIDER_DIR, json_filename)
    
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset not found: {json_path}")
    
    print(f"Loading Spider {split} dataset from: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
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
    
    print(f"Loaded {len(examples)} examples")
    return examples


def load_spider_tables() -> Dict[str, Any]:
    """Load database schema information from tables.json."""
    tables_path = os.path.join(SPIDER_DIR, 'tables.json')
    with open(tables_path, 'r', encoding='utf-8') as f:
        tables = json.load(f)
    return {table['db_id']: table for table in tables}


def get_database_schema(db_id: str) -> str:
    """Generate CREATE TABLE statements for a database with PK/FK."""
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
        if table_idx == -1:
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
        
        for col_idx, col_name, col_type in tables_columns[table_name]:
            schema_lines.append(f"  {col_name} {col_type},")
        
        # Primary Keys
        primary_keys = db_info.get('primary_keys', [])
        pk_columns = []
        for pk_idx in primary_keys:
            if pk_idx < len(column_names):
                pk_table_idx, pk_col_name = column_names[pk_idx]
                if pk_table_idx == table_idx:
                    pk_columns.append(pk_col_name)
        
        if pk_columns:
            schema_lines.append(f"  PRIMARY KEY ({', '.join(pk_columns)}),")
        
        # Foreign Keys
        for source_idx, target_idx in db_info.get('foreign_keys', []):
            if source_idx < len(column_names) and target_idx < len(column_names):
                source_table_idx, source_col = column_names[source_idx]
                target_table_idx, target_col = column_names[target_idx]
                if source_table_idx == table_idx:
                    target_table_name = table_names[target_table_idx]
                    schema_lines.append(f"  FOREIGN KEY ({source_col}) REFERENCES {target_table_name}({target_col}),")
        
        if schema_lines[-1].endswith(','):
            schema_lines[-1] = schema_lines[-1][:-1]
        schema_lines.append(")")
        schema_lines.append("")
    
    return "\n".join(schema_lines)


# ==============================================================================
# SECTION 3: MODEL LOADING (WITH OPTIONAL PEFT ADAPTER)
# ==============================================================================

def load_model(use_adapter: bool = True) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load the Qwen2.5-Coder-1.5B model with optional LoRA adapter.
    
    Args:
        use_adapter: If True and adapter exists, load the fine-tuned LoRA weights
        
    Returns:
        model: Base model or PeftModel (if adapter loaded)
        tokenizer: The tokenizer
    """
    print(f"Loading model: {MODEL_NAME}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    print(f"Loading model on {DEVICE} (this may take a moment for 1.5B model)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto" if DEVICE == "cuda" else None,
        trust_remote_code=True
    )
    
    if DEVICE == "cpu":
        model = model.to("cpu")
    
    # Check if we should load the fine-tuned LoRA adapter
    if use_adapter and os.path.exists(ADAPTER_PATH):
        print(f"\n🔧 Loading fine-tuned LoRA adapter from: {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
        print("✓ LoRA adapter loaded successfully!")
    elif use_adapter:
        print(f"\n⚠️  Adapter not found at: {ADAPTER_PATH}")
        print("   Running with base model only (zero-shot mode)")
    else:
        print("\n📌 Running with base model only (adapter loading disabled)")
    
    print(f"\nModel loaded successfully")
    return model, tokenizer


# ==============================================================================
# SECTION 4: IMPROVED SQL GENERATION
# ==============================================================================

def generate_sql_improved(model, tokenizer, query: str, schema_context: str) -> str:
    """
    Generate SQL with IMPROVED prompting and post-processing.
    
    Key improvements over baseline:
    1. Shorter max_new_tokens to prevent rambling
    2. Stop strings to halt at section markers
    3. Fuzzy column correction applied after generation
    """
    
    # Simplified prompt - less is more for larger model
    prompt = f"""### Instruction:
You are a text-to-SQL generator. Given the schema and question, output ONLY the SQL query.

### Schema:
{schema_context}

### Question:
{query}

### Response:
"""

    # Tokenize with truncation
    inputs = tokenizer(
        prompt, 
        return_tensors="pt", 
        truncation=True, 
        max_length=2048
    ).to(model.device)
    
    # Generate with better stop conditions
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,  # SHORTER than baseline to prevent rambling
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stop_strings=["###", "Question:", "Schema:"],  # Stop at next section
        tokenizer=tokenizer
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract SQL from response
    sql = extract_sql(response, prompt)
    
    # Clean the SQL
    sql = clean_sql(sql)
    
    # FUZZY COLUMN CORRECTION - Key improvement!
    sql = correct_sql_columns(sql, schema_context)
    
    return sql


# ==============================================================================
# SECTION 5: FUZZY COLUMN CORRECTION (Key Innovation)
# ==============================================================================

def correct_sql_columns(sql: str, schema: str) -> str:
    """
    Correct invalid column names in SQL using fuzzy matching.
    
    This is a KEY IMPROVEMENT that fixes hallucinated column names.
    
    Example:
        Input:  "SELECT petage FROM pets"
        Schema contains: "pet_age"
        Output: "SELECT pet_age FROM pets"
    
    How it works:
    1. Extract all valid column names from schema
    2. Extract all tokens from generated SQL
    3. For each token not in schema, find closest match
    4. Replace if similarity > 80%
    """
    
    # Step 1: Extract valid column and table names from schema
    schema_columns = set()
    schema_tables = set()
    
    # Pattern for table names: CREATE TABLE table_name
    table_pattern = r'^\s*CREATE\s+TABLE\s+([a-zA-Z0-9_]+)'
    
    # Pattern for column names: col_name type
    column_pattern = r'^\s*([a-zA-Z0-9_]+)\s+(?:INT|TEXT|number|text|DECIMAL|DATE|REAL|INTEGER|double|float)'
    
    for line in schema.split('\n'):
        # Check for table
        table_match = re.match(table_pattern, line, re.IGNORECASE)
        if table_match:
            schema_tables.add(table_match.group(1).lower())
            continue
        
        # Check for column
        col_match = re.match(column_pattern, line, re.IGNORECASE)
        if col_match:
            schema_columns.add(col_match.group(1).lower())
    
    if not schema_columns:
        return sql  # No columns found, return as-is
    
    # Step 2: Extract tokens from SQL
    tokens = re.findall(r'\b[a-zA-Z0-9_]+\b', sql)
    
    # SQL keywords to skip
    keywords = {
        'select', 'from', 'where', 'and', 'or', 'order', 'by', 'group',
        'having', 'limit', 'join', 'on', 'as', 'asc', 'desc', 'count',
        'avg', 'sum', 'min', 'max', 'distinct', 'between', 'like', 'in',
        'not', 'null', 'is', 'true', 'false', 'case', 'when', 'then',
        'else', 'end', 'cast', 'inner', 'left', 'right', 'outer', 'full',
        'create', 'table', 'primary', 'key', 'foreign', 'references',
        'int', 'text', 'date', 'year', 'month', 'day'
    }
    
    # Step 3: Find replacements
    replacements = {}
    
    for token in tokens:
        lower_token = token.lower()
        
        # Skip keywords, numbers, aliases (T1, T2), valid columns/tables
        if (lower_token in keywords or 
            token.isdigit() or 
            re.match(r'^t\d+$', lower_token) or
            lower_token in schema_columns or 
            lower_token in schema_tables):
            continue
        
        # Find closest match using difflib
        matches = difflib.get_close_matches(
            lower_token, 
            schema_columns, 
            n=1, 
            cutoff=0.8  # 80% similarity threshold
        )
        
        if matches:
            best_match = matches[0]
            # Only replace if length difference is small
            if abs(len(best_match) - len(lower_token)) <= 3:
                replacements[token] = best_match
    
    # Step 4: Apply replacements (longer first to avoid partial matches)
    for bad_col, good_col in sorted(replacements.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = r'\b' + re.escape(bad_col) + r'\b'
        sql = re.sub(pattern, good_col, sql)
    
    return sql


def extract_sql(response: str, original_prompt: str) -> str:
    """Extract SQL from model response."""
    
    # Remove the prompt from response
    if "### Response:" in response:
        response = response.split("### Response:")[-1]
    elif original_prompt in response:
        response = response.split(original_prompt)[-1]
    
    # Remove markdown code blocks
    if "```sql" in response:
        response = response.split("```sql")[1].split("```")[0]
    elif "```" in response:
        parts = response.split("```")
        if len(parts) >= 2:
            response = parts[1]
    
    return response.strip()


def clean_sql(sql: str) -> str:
    """Clean and validate SQL query."""
    
    # IMPORTANT: Stop at chat continuation patterns (from fine-tuned model)
    # The model sometimes continues with "Human:" or "Assistant:" after the SQL
    chat_stop_patterns = [
        r'Human:.*',
        r'Assistant:.*',
        r'<\|im_start\|>.*',
        r'<\|im_end\|>.*',
    ]
    
    for pattern in chat_stop_patterns:
        sql = re.split(pattern, sql, flags=re.IGNORECASE)[0]
    
    # Remove common explanation patterns
    explanation_patterns = [
        r'\bTo generate\b.*',
        r'\bThis query\b.*',
        r'\bThe query\b.*',
        r'\bExplanation:.*',
        r'\bNote:.*',
        r'\bHere\'s.*',
    ]
    
    for pattern in explanation_patterns:
        sql = re.split(pattern, sql, flags=re.IGNORECASE)[0]
    
    # Process line by line
    lines = sql.split('\n')
    sql_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Stop at explanation lines
        if any(line.lower().startswith(word) for word in 
               ['to ', 'this ', 'the ', 'note', 'explanation', 'here', 'human', 'assistant']):
            break
        
        # Stop at second SELECT (likely explanation example)
        if len(sql_lines) > 0 and line.upper().startswith('SELECT'):
            break
        
        # Skip comments
        if not line.startswith('--') and not line.startswith('#'):
            sql_lines.append(line)
    
    sql = ' '.join(sql_lines)
    
    # Remove quotes around entire query
    sql = sql.strip().strip('"').strip("'")
    
    # Keep only first statement
    if ';' in sql:
        sql = sql.split(';')[0].strip()
    
    # Normalize whitespace
    sql = re.sub(r'\s+', ' ', sql).strip()
    
    return sql


# ==============================================================================
# SECTION 6: EVALUATION METRICS (Same as baseline)
# ==============================================================================

def normalize_sql(sql: str) -> str:
    """Normalize SQL for comparison."""
    sql = sql.lower()
    sql = re.sub(r'\s+', ' ', sql)
    sql = sql.rstrip(';').strip()
    return sql


def compute_exact_match(predicted: str, reference: str) -> bool:
    """Check if predicted SQL exactly matches reference."""
    return normalize_sql(predicted) == normalize_sql(reference)


def execute_sql(sql: str, db_path: str) -> Tuple[bool, Optional[list], Optional[str]]:
    """Execute SQL query on SQLite database."""
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
    """Check if predicted and reference SQL return same results."""
    pred_success, pred_results, _ = execute_sql(predicted, db_path)
    if not pred_success:
        return False
    
    ref_success, ref_results, _ = execute_sql(reference, db_path)
    if not ref_success:
        return False
    
    try:
        pred_set = set(tuple(row) for row in pred_results)
        ref_set = set(tuple(row) for row in ref_results)
        return pred_set == ref_set
    except TypeError:
        return sorted(pred_results) == sorted(ref_results)


# ==============================================================================
# SECTION 7: MAIN EVALUATION LOOP
# ==============================================================================

def main():
    """
    Main evaluation function for IMPROVED model.
    
    Same flow as baseline but uses:
    - Larger 1.5B model
    - Optional fine-tuned LoRA adapter (if available)
    - Improved generation with stop strings
    - Fuzzy column correction
    """
    
    parser = argparse.ArgumentParser(description="Improved Text-to-SQL Evaluation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of examples")
    parser.add_argument("--complex-only", action="store_true", help="Evaluate only complex queries")
    parser.add_argument("--no-adapter", action="store_true", help="Don't load LoRA adapter (use base model only)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("IMPROVED TEXT-TO-SQL EVALUATION")
    print("=" * 60)
    
    # Load model (with adapter by default if available)
    use_adapter = not args.no_adapter
    model, tokenizer = load_model(use_adapter=use_adapter)
    
    # Load dataset
    examples = load_spider_dataset(split='dev', limit=args.limit if not args.complex_only else None)
    
    if args.complex_only:
        examples = [e for e in examples if e.complexity == 'complex']
        if args.limit:
            examples = examples[:args.limit]
    
    print(f"\nEvaluating {len(examples)} examples...\n")
    
    # Evaluate
    results = []
    exact_matches = 0
    execution_matches = 0
    
    for example in tqdm(examples, desc="Evaluating"):
        schema = get_database_schema(example.db_id)
        
        try:
            generated_sql = generate_sql_improved(model, tokenizer, example.question, schema)
        except Exception as e:
            print(f"Error: {e}")
            generated_sql = "SELECT * FROM error"
        
        exact_match = compute_exact_match(generated_sql, example.query)
        execution_match = False
        
        if example.db_path and os.path.exists(example.db_path):
            execution_match = compute_execution_accuracy(
                generated_sql, example.query, example.db_path
            )
        
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
    
    # Print summary
    total = len(results)
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY (IMPROVED)")
    print("=" * 60)
    print(f"Total Examples: {total}")
    print(f"Exact Match Accuracy: {exact_matches}/{total} ({100*exact_matches/total:.2f}%)")
    print(f"Execution Accuracy: {execution_matches}/{total} ({100*execution_matches/total:.2f}%)")
    print("=" * 60)
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_file = os.path.join(OUTPUT_DIR, 'improved_results.json')
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

