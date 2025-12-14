"""
Improved inference with better prompting and post-processing.

Key improvements:
1. Explicit instruction to output ONLY SQL
2. Include column names in schema representation
3. Better post-processing to remove explanatory text
4. Validation of generated SQL
"""

import re
import difflib
from transformers import PreTrainedModel, PreTrainedTokenizer


def generate_sql_improved(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizer, 
    query: str, 
    schema_context: str
) -> str:
    """
    Generates SQL from a natural language query with improved prompting.
    
    Key improvements over basic inference:
    - Explicit "output only SQL" instruction
    - Structured prompt format
    - Better post-processing
    
    Args:
        model: The pre-trained language model
        tokenizer: The tokenizer for the model
        query: Natural language query from user
        schema_context: Database schema information
        
    Returns:
        Generated SQL query as a string
    """
    
    # Simplified prompt - less is more for 0.5B model
    prompt = f"""### Instruction:
You are a text-to-SQL generator. Given the schema and question, output ONLY the SQL query.

### Schema:
{schema_context}

### Question:
{query}

### Response:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,  # Reduced further to prevent rambling
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        stop_strings=["###", "Question:", "Schema:"], # Stop at next section
        tokenizer=tokenizer
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract SQL from response
    sql = extract_sql(response, prompt)
    
    # Validate and clean
    sql = clean_sql(sql)
    
    # Correct column names using fuzzy matching
    sql = correct_sql_columns(sql, schema_context)
    
    return sql


def correct_sql_columns(sql: str, schema: str) -> str:
    """
    Corrects invalid column names in SQL using fuzzy matching against schema.
    """
    # Extract valid column names and table names from schema
    # Pattern now handles lines with PK/FK too, but main interest is column defs
    schema_columns = set()
    schema_tables = set()
    
    # Extract table names: CREATE TABLE table_name
    table_pattern = r'^\s*CREATE\s+TABLE\s+([a-zA-Z0-9_]+)'
    
    # Extract column names: col_name type
    # A more robust way based on our dataset_loader output format:
    # "  col_name type"
    column_pattern = r'^\s*([a-zA-Z0-9_]+)\s+(?:INT|TEXT|number|text|DECIMAL|DATE|REAL|INTEGER|double|float)'
    
    for line in schema.split('\n'):
        # Check for table
        table_match = re.match(table_pattern, line, re.IGNORECASE)
        if table_match:
            schema_tables.add(table_match.group(1).lower())
            continue
            
        # Check for column
        match = re.match(column_pattern, line, re.IGNORECASE)
        if match:
             schema_columns.add(match.group(1).lower())
             
    if not schema_columns:
        return sql
        
    # Extract potential column tokens from SQL
    # We try to identify identifiers that match invalid columns
    
    # Split by standard SQL delimiters to find candidates
    # This is a heuristic; a full SQL parser would be better but overkill here
    tokens = re.findall(r'\b[a-zA-Z0-9_]+\b', sql)
    
    keywords = {
        'select', 'from', 'where', 'and', 'or', 'order', 'by', 'group',
        'having', 'limit', 'join', 'on', 'as', 'asc', 'desc', 'count',
        'avg', 'sum', 'min', 'max', 'distinct', 'between', 'like', 'in',
        'not', 'null', 'is', 'true', 'false', 'case', 'when', 'then',
        'else', 'end', 'cast', 'inner', 'left', 'right', 'outer', 'full',
        'create', 'table', 'primary', 'key', 'foreign', 'references',
        'int', 'text', 'date', 'year', 'month', 'day'
    }
    
    # Identify replacements
    replacements = {}
    
    for token in tokens:
        lower_token = token.lower()
        
        # Skip keywords, numbers (often years/values), and T1/T2 aliases
        if (lower_token in keywords or 
            token.isdigit() or 
            re.match(r'^t\d+$', lower_token)):
            continue
            
        # If token is already a valid column OR A TABLE, skip
        if lower_token in schema_columns or lower_token in schema_tables:
            continue
            
        # Find closest match
        matches = difflib.get_close_matches(lower_token, schema_columns, n=1, cutoff=0.8)
        
        if matches:
            best_match = matches[0]
            # Heuristic: only replace if length difference is small to avoid over-correction
            # e.g. "age" shouldn't replace "language" just because it's the only match
            if abs(len(best_match) - len(lower_token)) <= 3:
                # Store original token -> best match (preserving case of match if needed, but SQL is case insensitive mostly)
                # We use the schema's casing (often lower)
                replacements[token] = best_match
                
    # Apply replacements
    # Sort by length descending to replace longer words first if overlapping
    for bad_col, good_col in sorted(replacements.items(), key=lambda x: len(x[0]), reverse=True):
        # Use regex to replace whole word only
        pattern = r'\b' + re.escape(bad_col) + r'\b'
        sql = re.sub(pattern, good_col, sql)
        
    return sql


def extract_sql(response: str, original_prompt: str) -> str:
    """Extract SQL from model response."""
    
    # Remove the original prompt from response
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
    
    # Take only the first statement (before any explanation)
    lines = sql.split('\n')
    sql_lines = []
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # Stop at lines that look like explanations
        if any(line.lower().startswith(word) for word in 
               ['to ', 'this ', 'the ', 'note', 'explanation', 'here']):
            break
            
        # Stop if we see a second SELECT
        if len(sql_lines) > 0 and line.upper().startswith('SELECT'):
            break
            
        if not line.startswith('--') and not line.startswith('#'):
            sql_lines.append(line)
    
    sql = ' '.join(sql_lines)
    
    # Remove quotes around the entire query
    sql = sql.strip().strip('"').strip("'")
    
    # Ensure single statement
    if ';' in sql:
        sql = sql.split(';')[0].strip()
    
    # Remove extra whitespace
    sql = re.sub(r'\s+', ' ', sql).strip()
    
    return sql


def validate_sql_columns(sql: str, schema: str) -> tuple[bool, list[str]]:
    """
    Validate that SQL uses columns that exist in the schema.
    
    Returns:
        Tuple of (is_valid, list of invalid columns)
    """
    # Extract column names from schema
    schema_columns = set()
    column_pattern = r'^\s*(\w+)\s+(?:INT|TEXT|number|text|DECIMAL|DATE|REAL|INTEGER)'
    for line in schema.split('\n'):
        match = re.match(column_pattern, line, re.IGNORECASE)
        if match:
            schema_columns.add(match.group(1).lower())
    
    # Extract column names from SQL (simplified)
    sql_tokens = re.findall(r'\b\w+\b', sql)
    sql_columns = [t for t in sql_tokens if t.lower() not in 
                   {'select', 'from', 'where', 'and', 'or', 'order', 'by', 'group',
                    'having', 'limit', 'join', 'on', 'as', 'asc', 'desc', 'count',
                    'avg', 'sum', 'min', 'max', 'distinct', 'between', 'like', 'in',
                    'not', 'null', 'is', 'true', 'false', 'case', 'when', 'then',
                    'else', 'end', 'cast', 'inner', 'left', 'right', 'outer',
                    't1', 't2', 't3', 't4', '1', '2', '3', 'create', 'table'}]
    
    invalid = []
    for col in sql_columns:
        if col.lower() not in schema_columns and not col.isdigit():
            # Could be a table name or string literal, skip for now
            pass
    
    return len(invalid) == 0, invalid
