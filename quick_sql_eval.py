"""
Quick SQL accuracy evaluation - compares generated vs gold SQL from Spider dataset
Uses cached results or generates in batches with progress tracking
"""

import json
import os
from sql_accuracy_matrix import SQLAccuracyMatrix
from Data.Spider.SpiderLoader import SpiderLoader
from Core.Chunk.ChunkFactory import ChunkFactory
from Core.Graph.GraphBuilder import GraphBuilderFactory
from Core.Retriever.SchemaRetriever import SchemaRetriever
from Core.LLM.SQLGenerator import SQLGenerator
from Core.LLM.OllamaLLMClient import OllamaLLMClient
import logging

logging.basicConfig(level=logging.ERROR)  # Suppress verbose output


def quick_evaluate(max_examples=50, use_cache=True, cache_file='spider_sql_cache.json'):
    """
    Quick SQL accuracy evaluation with caching
    """
    print("\n" + "=" * 100)
    print("🚀 QUICK SQL ACCURACY EVALUATION - SPIDER DATASET")
    print("=" * 100)
    
    # Load cached results if available
    cached_results = []
    if use_cache and os.path.exists(cache_file):
        print(f"📦 Loading cached results from {cache_file}...")
        with open(cache_file, 'r') as f:
            cached_results = json.load(f)
        print(f"✅ Loaded {len(cached_results)} cached query pairs\n")
    
    # If we need to generate more
    if len(cached_results) < max_examples:
        print(f"🔨 Generating {max_examples - len(cached_results)} SQL queries...")
        print("(This may take a while as it requires LLM calls)\n")
        
        # Initialize components
        loader = SpiderLoader('Data/Spider')
        fac = ChunkFactory()
        builder_factory = GraphBuilderFactory()
        llm = OllamaLLMClient()
        gen = SQLGenerator(llm)
        
        # Suppress verbose logging
        logging.getLogger('Core.Graph.GraphBuilder').setLevel(logging.ERROR)
        
        # Load examples
        examples = list(loader.load_examples(split='dev'))
        start_idx = len(cached_results)
        
        # Generate missing queries
        for idx, ex in enumerate(examples[start_idx:start_idx + (max_examples - len(cached_results))], start_idx + 1):
            try:
                print(f"  [{idx}/{max_examples}] {ex.get('db_id', 'unknown')}... ", end="", flush=True)
                
                # Build graph and retrieve
                chunks = fac.schema_to_chunks(ex['db_schema'])
                G = builder_factory.build_graph('schema', chunks)
                _, sub = SchemaRetriever(G).retrieve(ex['question'])
                
                # Generate SQL
                generated_sql = gen.generate_sql(ex['question'], sub)
                
                # Store result
                cached_results.append({
                    'db_id': ex.get('db_id', 'unknown'),
                    'question': ex['question'],
                    'gold_sql': ex['query'],
                    'generated_sql': generated_sql
                })
                
                print("✓")
                
                # Save cache every 5 queries
                if idx % 5 == 0:
                    with open(cache_file, 'w') as f:
                        json.dump(cached_results, f, indent=2)
                    print(f"    💾 Cache saved ({len(cached_results)} queries)")
                
            except Exception as e:
                print(f"✗ ({str(e)[:40]})")
                continue
        
        # Final save
        with open(cache_file, 'w') as f:
            json.dump(cached_results, f, indent=2)
        print(f"\n✅ Cache saved to {cache_file}\n")
    
    # Evaluate accuracy
    print("=" * 100)
    print("📊 EVALUATING SQL ACCURACY")
    print("=" * 100 + "\n")
    
    matrix = SQLAccuracyMatrix()
    
    for result in cached_results[:max_examples]:
        matrix.evaluate_query(
            generated=result['generated_sql'],
            gold=result['gold_sql'],
            question=result['question'],
            db_id=result['db_id']
        )
    
    # Print report
    matrix.print_detailed_report(show_individual=False)
    
    # Export detailed results
    matrix.export_results('spider_sql_accuracy_results.json')
    
    return matrix


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Quick SQL accuracy evaluation')
    parser.add_argument('--max-examples', type=int, default=50,
                       help='Maximum number of examples to evaluate')
    parser.add_argument('--no-cache', action='store_true',
                       help='Disable caching (regenerate all queries)')
    parser.add_argument('--cache-file', default='spider_sql_cache.json',
                       help='Cache file path')
    parser.add_argument('--show-details', action='store_true',
                       help='Show individual query details')
    
    args = parser.parse_args()
    
    matrix = quick_evaluate(
        max_examples=args.max_examples,
        use_cache=not args.no_cache,
        cache_file=args.cache_file
    )
    
    if args.show_details:
        print("\n" + "=" * 100)
        print("📝 TOP 10 QUERY DETAILS")
        print("=" * 100)
        
        for i, result in enumerate(matrix.results[:10], 1):
            print(f"\n{'─' * 100}")
            print(f"Query {i}: {result['accuracy_level']} "
                  f"(Similarity: {result['similarity_score']:.1%})")
            print(f"{'─' * 100}")
            print(f"Database: {result['db_id']}")
            print(f"Question: {result['question']}")
            print(f"\nGold SQL:\n  {result['gold_sql']}")
            print(f"\nGenerated SQL:\n  {result['generated_sql']}")
            
            # Show mismatches
            mismatches = [k for k, v in result['component_comparison'].items() if not v]
            if mismatches:
                print(f"\n⚠️  Mismatches: {', '.join(mismatches)}")
