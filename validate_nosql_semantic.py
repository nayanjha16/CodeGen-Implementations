"""
Semantic NoSQL Validation Evaluation
Validates generated NoSQL based on SQL logic correctness
"""

import json
import os
from nosql_semantic_validator import NoSQLSemanticValidator
from Core.LLM.SQLGenerator import SQLGenerator
from Core.LLM.OllamaLLMClient import OllamaLLMClient
import logging

logging.basicConfig(level=logging.ERROR)


def load_sql_queries(sql_file, max_queries=None):
    """Load SQL queries from file"""
    with open(sql_file, 'r') as f:
        queries = [line.strip().split('\t')[0].strip() for line in f if line.strip()]
    
    if max_queries:
        queries = queries[:max_queries]
    
    return queries


def validate_nosql_generation(sql_file='Data/Spider/noSQL2SQL/train_gold (1) (1).sql',
                              max_examples=50,
                              export_file='nosql_validation_results.json',
                              use_cache=True,
                              cache_file='nosql_generation_cache.json'):
    """
    Generate and validate NoSQL translations using semantic validation
    """
    
    print("\n" + "=" * 100)
    print("🔍 NOSQL SEMANTIC VALIDATION")
    print("=" * 100)
    print(f"SQL Data: {sql_file}")
    print(f"Max Examples: {max_examples}")
    print(f"Model: qwen2.5-coder:3b")
    print(f"Validation: Semantic logic checking (not reference comparison)")
    print("=" * 100 + "\n")
    
    # Load SQL queries
    print("Loading SQL queries...")
    sql_queries = load_sql_queries(sql_file, max_examples)
    print(f"✅ Loaded {len(sql_queries)} SQL queries\n")
    
    # Check cache
    cached_results = []
    if use_cache and os.path.exists(cache_file):
        print(f"📦 Loading cached results from {cache_file}...")
        with open(cache_file, 'r') as f:
            cached_data = json.load(f)
            # Extract only sql and generated_nosql for validation
            cached_results = [{'sql': r['sql'], 'generated_nosql': r['generated_nosql']} 
                            for r in cached_data]
        print(f"✅ Loaded {len(cached_results)} cached results\n")
    
    # Generate missing NoSQL queries
    if len(cached_results) < max_examples:
        print(f"🔨 Generating {max_examples - len(cached_results)} NoSQL queries...")
        print("(This may take a while as it requires LLM calls)\n")
        
        # Initialize LLM with qwen model
        llm = OllamaLLMClient()
        gen = SQLGenerator(llm, nosql_model="qwen2.5-coder:3b")
        
        start_idx = len(cached_results)
        full_cache = []
        
        # Load existing full cache if exists
        if os.path.exists(cache_file):
            with open(cache_file, 'r') as f:
                full_cache = json.load(f)
        
        for idx in range(start_idx, max_examples):
            try:
                sql_query = sql_queries[idx]
                
                print(f"  [{idx+1}/{max_examples}] Generating... ", end="", flush=True)
                
                # Generate NoSQL from SQL
                generated_nosql = gen.generate_nosql(sql_query, subgraph=None, target='mongodb')
                
                # Add to results
                cached_results.append({
                    'sql': sql_query,
                    'generated_nosql': generated_nosql
                })
                
                # Add to full cache (with gold_nosql placeholder)
                full_cache.append({
                    'sql': sql_query,
                    'generated_nosql': generated_nosql,
                    'gold_nosql': None  # Not using for validation
                })
                
                print("✓")
                
                # Save cache every 5 queries
                if (idx + 1) % 5 == 0:
                    with open(cache_file, 'w') as f:
                        json.dump(full_cache, f, indent=2)
                    print(f"    💾 Cache saved ({len(full_cache)} queries)")
                
            except Exception as e:
                print(f"✗ ({str(e)[:40]})")
                continue
        
        # Final save
        with open(cache_file, 'w') as f:
            json.dump(full_cache, f, indent=2)
        print(f"\n✅ Cache saved to {cache_file}\n")
    
    # Perform semantic validation
    print("=" * 100)
    print("📊 PERFORMING SEMANTIC VALIDATION")
    print("=" * 100 + "\n")
    
    validator = NoSQLSemanticValidator()
    
    for i, result in enumerate(cached_results[:max_examples]):
        validator.validate_translation(
            sql=result['sql'],
            nosql=result['generated_nosql'],
            index=i+1
        )
    
    # Print report
    validator.print_detailed_report(show_individual=False)
    
    # Export detailed results
    if export_file:
        validator.export_results(export_file)
    
    return validator


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Semantic NoSQL validation')
    parser.add_argument('--sql-file',
                       default='Data/Spider/noSQL2SQL/train_gold (1) (1).sql',
                       help='Path to SQL file')
    parser.add_argument('--max-examples', type=int, default=50,
                       help='Maximum number of examples to validate')
    parser.add_argument('--export', default='nosql_validation_results.json',
                       help='Output file for results')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching (regenerate all queries)')
    parser.add_argument('--cache-file', default='nosql_generation_cache.json',
                       help='Cache file path')
    parser.add_argument('--show-details', action='store_true',
                       help='Show individual query details')
    
    args = parser.parse_args()
    
    validator = validate_nosql_generation(
        sql_file=args.sql_file,
        max_examples=args.max_examples,
        export_file=args.export,
        use_cache=not args.no_cache,
        cache_file=args.cache_file
    )
    
    if args.show_details:
        print("\n" + "=" * 100)
        print("📝 TOP 10 QUERY DETAILS")
        print("=" * 100)
        
        for i, result in enumerate(validator.results[:10], 1):
            print(f"\n{'─' * 100}")
            print(f"Query {i}: {result['validation_level']} ({result['validation_percentage']:.0%})")
            print(f"{'─' * 100}")
            print(f"SQL: {result['sql']}")
            print(f"\nGenerated NoSQL:\n  {result['nosql'][:300]}...")
            
            print(f"\n✅ Passed Validations:")
            for val, passed in result['validations'].items():
                if passed:
                    print(f"   • {val}")
            
            if result['issues']:
                print(f"\n❌ Issues Found:")
                for issue in result['issues']:
                    print(f"   • {issue}")
