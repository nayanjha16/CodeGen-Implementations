"""
Quick script to run evaluation and save results to JSON.
"""

import sys
import os
import json

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.dataset_loader import load_spider_dataset, get_database_schema
from src.model_loader import load_model
from src.inference import generate_sql
from src.evaluate import compute_exact_match, compute_execution_accuracy, compute_metrics

def main():
    print("=" * 80)
    print("Text-to-SQL Model Evaluation on Spider Dataset")
    print("=" * 80)
    print()
    
    # Load dataset
    print("Loading Spider dev dataset...")
    examples = load_spider_dataset(split='dev', limit=20)
    print(f"Loaded {len(examples)} examples")
    print()
    
    # Load model
    print("Loading model and tokenizer...")
    model, tokenizer = load_model()
    print("[OK] Model loaded successfully")
    print()
    
    # Run evaluation
    print("Running evaluation...")
    print("-" * 80)
    
    exact_matches = []
    execution_matches = []
    results = []
    
    for idx, example in enumerate(examples):
        print(f"Evaluating example {idx + 1}/{len(examples)}...", end='\r')
        
        # Get database schema
        try:
            schema = get_database_schema(example.db_id)
        except Exception as e:
            exact_matches.append(False)
            execution_matches.append(False)
            results.append({
                'question': example.question,
                'db_id': example.db_id,
                'reference': example.query,
                'predicted': None,
                'error': f"Schema load failed: {e}",
                'exact_match': False,
                'execution_match': False
            })
            continue
        
        # Generate SQL
        try:
            predicted_sql = generate_sql(
                model=model,
                tokenizer=tokenizer,
                query=example.question,
                schema_context=schema
            )
        except Exception as e:
            exact_matches.append(False)
            execution_matches.append(False)
            results.append({
                'question': example.question,
                'db_id': example.db_id,
                'reference': example.query,
                'predicted': None,
                'error': f"Generation failed: {e}",
                'exact_match': False,
                'execution_match': False
            })
            continue
        
        # Compute exact match
        exact_match = compute_exact_match(predicted_sql, example.query)
        exact_matches.append(exact_match)
        
        # Compute execution accuracy
        if example.db_path and os.path.exists(example.db_path):
            execution_match = compute_execution_accuracy(
                predicted_sql,
                example.query,
                example.db_path,
                verbose=False
            )
        else:
            execution_match = False
        
        execution_matches.append(execution_match)
        
        results.append({
            'question': example.question,
            'db_id': example.db_id,
            'reference': example.query,
            'predicted': predicted_sql,
            'exact_match': exact_match,
            'execution_match': execution_match
        })
    
    print()
    print()
    
    # Compute and display metrics
    print("=" * 80)
    print("Evaluation Results")
    print("=" * 80)
    
    metrics = compute_metrics(exact_matches, execution_matches)
    
    print(f"Total Examples:        {metrics['total_examples']}")
    print(f"Exact Match Count:     {metrics['exact_match_count']}")
    print(f"Exact Match Accuracy:  {metrics['exact_match_accuracy']:.2f}%")
    print(f"Execution Count:       {metrics['execution_count']}")
    print(f"Execution Accuracy:    {metrics['execution_accuracy']:.2f}%")
    print("=" * 80)
    
    # Save results to JSON
    output = {
        'metrics': metrics,
        'results': results
    }
    
    with open('evaluation_results.json', 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    print()
    print("Results saved to: evaluation_results.json")

if __name__ == "__main__":
    main()
