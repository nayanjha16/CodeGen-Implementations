"""
Dataset loader for Spider benchmark dataset.

This module provides utilities to load and parse the Spider dataset
for evaluation purposes.
"""

import json
import os
from typing import List, Dict, Any, Optional


class SpiderExample:
    """
    Represents a single example from the Spider dataset.
    """
    
    def __init__(
        self,
        question: str,
        query: str,
        db_id: str,
        db_path: Optional[str] = None
    ):
        """
        Initialize a Spider example.
        
        Args:
            question: Natural language question
            query: Ground truth SQL query
            db_id: Database identifier
            db_path: Path to the SQLite database file
        """
        self.question = question
        self.query = query
        self.db_id = db_id
        self.db_path = db_path
        self.complexity = self._calculate_complexity()

    def _calculate_complexity(self) -> str:
        sql = self.query.lower()
        if "select" in sql and sql.count("select") > 1:
            return "complex" # Nested queries
        if "join" in sql or "union" in sql or "intersect" in sql or "except" in sql:
            return "complex"
        return "simple"
    
    def __repr__(self) -> str:
        return f"SpiderExample(db='{self.db_id}', complexity='{self.complexity}', question='{self.question[:50]}...')"


def load_spider_dataset(
    split: str = 'dev',
    spider_dir: Optional[str] = None,
    limit: Optional[int] = None,
    smart_filter: bool = False
) -> List[SpiderExample]:
    """
    Load the Spider dataset from JSON files.
    
    Args:
        split: Dataset split to load ('train' or 'dev')
        spider_dir: Path to spider directory. If None, uses default location.
        limit: Maximum number of examples to load (for testing)
        smart_filter: If True, returns 100 examples with last 25 being complex (overrides normal limit logic if set)
        
    Returns:
        List of SpiderExample objects
        
    Raises:
        FileNotFoundError: If dataset files are not found
        ValueError: If split is invalid
    """
    # Validate split
    if split not in ['train', 'dev']:
        raise ValueError(f"Invalid split: {split}. Must be 'train' or 'dev'")
    
    # Determine spider directory
    if spider_dir is None:
        # Default: data/spider relative to project root
        current_dir = os.path.dirname(__file__)
        spider_dir = os.path.abspath(os.path.join(current_dir, '..', 'data', 'spider'))
    
    # Check if directory exists
    if not os.path.exists(spider_dir):
        raise FileNotFoundError(
            f"Spider directory not found: {spider_dir}\n"
            f"Please run 'python scripts/download_spider.py' first."
        )
    
    # Load JSON file
    json_filename = f'{split}.json'
    if split == 'train' and not os.path.exists(os.path.join(spider_dir, json_filename)):
        json_filename = 'train_spider.json'
        
    json_path = os.path.join(spider_dir, json_filename)
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Dataset file not found: {json_path}")
    
    print(f"Loading Spider {split} dataset from: {json_path}")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Parse examples
    examples = []
    database_dir = os.path.join(spider_dir, 'database')
    
    all_examples = []
    for idx, item in enumerate(data):
        db_id = item['db_id']
        db_path = os.path.join(database_dir, db_id, f'{db_id}.sqlite')
        
        example = SpiderExample(
            question=item['question'],
            query=item['query'],
            db_id=db_id,
            db_path=db_path if os.path.exists(db_path) else None
        )
        all_examples.append(example)

    if smart_filter:
        total_limit = limit if limit is not None else 100
        # 25% complex, rest simple
        n_complex = max(1, int(total_limit * 0.25))
        n_simple = total_limit - n_complex
        
        print(f"Applying smart filter: {total_limit} examples ({n_simple} mixed, {n_complex} complex)...")
        complex_examples = [e for e in all_examples if e.complexity == 'complex']
        simple_examples = [e for e in all_examples if e.complexity == 'simple']
        
        # Take required amount
        selected_simple = simple_examples[:n_simple]
        selected_complex = complex_examples[:n_complex]
        
        examples = selected_simple + selected_complex
        print(f"Selected {len(selected_simple)} simple/mixed and {len(selected_complex)} complex examples.")
    else:
        # Standard limit logic
        if limit is not None:
            examples = all_examples[:limit]
        else:
            examples = all_examples
    
    print(f"Loaded {len(examples)} examples from Spider {split} set")
    
    return examples


def load_spider_tables(spider_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Load the Spider tables.json file containing database schemas.
    
    Args:
        spider_dir: Path to spider directory. If None, uses default location.
        
    Returns:
        Dictionary mapping database IDs to schema information
        
    Raises:
        FileNotFoundError: If tables.json is not found
    """
    # Determine spider directory
    if spider_dir is None:
        current_dir = os.path.dirname(__file__)
        spider_dir = os.path.abspath(os.path.join(current_dir, '..', 'data', 'spider'))
    
    tables_path = os.path.join(spider_dir, 'tables.json')
    
    if not os.path.exists(tables_path):
        raise FileNotFoundError(f"Tables file not found: {tables_path}")
    
    with open(tables_path, 'r', encoding='utf-8') as f:
        tables = json.load(f)
    
    # Convert list to dictionary for easier lookup
    tables_dict = {table['db_id']: table for table in tables}
    
    return tables_dict


def get_database_schema(db_id: str, spider_dir: Optional[str] = None) -> str:
    """
    Get the CREATE TABLE schema for a specific database.
    
    Args:
        db_id: Database identifier
        spider_dir: Path to spider directory. If None, uses default location.
        
    Returns:
        String containing CREATE TABLE statements
        
    Raises:
        FileNotFoundError: If database schema is not found
        KeyError: If database ID is not found in tables.json
    """
    tables = load_spider_tables(spider_dir)
    
    if db_id not in tables:
        raise KeyError(f"Database '{db_id}' not found in tables.json")
    
    db_info = tables[db_id]
    
    # Build CREATE TABLE statements
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
        tables_columns[table_name].append(f"  {col_name} {col_type}")
    
    # Generate CREATE TABLE statements
    for table_idx, (table_name, columns) in enumerate(tables_columns.items()):
        schema_lines.append(f"CREATE TABLE {table_name} (")
        
        # Add columns
        schema_lines.extend(columns)
        
        # Add Primary Keys
        primary_keys = db_info.get('primary_keys', [])
        pk_columns = []
        for pk_idx in primary_keys:
            # pk_idx is global column index
            # Find which table and local column it belongs to
            # We need to look up column_names with this index
            if pk_idx < len(column_names):
                pk_table_idx, pk_col_name = column_names[pk_idx]
                if pk_table_idx == table_idx:
                    pk_columns.append(pk_col_name)
                    
        if pk_columns:
            schema_lines.append(f",\n  PRIMARY KEY ({', '.join(pk_columns)})")
            
        # Add Foreign Keys
        foreign_keys = db_info.get('foreign_keys', [])
        for fk_pair in foreign_keys:
            # pair is [source_col_idx, target_col_idx]
            source_idx, target_idx = fk_pair
            
            if source_idx < len(column_names) and target_idx < len(column_names):
                source_table_idx, source_col = column_names[source_idx]
                target_table_idx, target_col = column_names[target_idx]
                
                # If this FK belongs to current table
                if source_table_idx == table_idx:
                    target_table_name = table_names[target_table_idx]
                    schema_lines.append(f",\n  FOREIGN KEY ({source_col}) REFERENCES {target_table_name}({target_col})")

        schema_lines.append(")")
        schema_lines.append("")
    
    return "\n".join(schema_lines)
