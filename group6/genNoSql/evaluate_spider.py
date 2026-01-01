"""
Evaluation Script for Spider Dev MongoDB Dataset
Runs SMART pipeline on subset of Spider dev queries and evaluates performance
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from smart.smart_pipeline import create_smart_framework
from evaluation.evaluator import Evaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_spider_data(file_path: str, max_samples: int = None) -> List[Dict[str, Any]]:
    """
    Load Spider dev MongoDB data
    
    Args:
        file_path: Path to spider_dev_mongo.json
        max_samples: Maximum number of samples to load (None for all)
    
    Returns:
        List of test cases
    """
    logger.info(f"Loading data from {file_path}")
    
    with open(file_path) as f:
        data = json.load(f)
    
    if max_samples:
        data = data[:max_samples]
        logger.info(f"Limited to {max_samples} samples")
    
    logger.info(f"Loaded {len(data)} test cases")
    return data


def extract_schemas_from_data(data: List[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Extract database schemas from MongoDB queries
    
    Args:
        data: List of test cases
    
    Returns:
        Dict mapping database names to collections
    """
    schemas = {}
    
    for item in data:
        db_id = item.get('db_id')
        mongo_query = item.get('mongo_query', '')
        
        if not mongo_query or db_id not in schemas:
            schemas[db_id] = set()
        
        # Extract collection name from query (e.g., "db.collection.find" -> "collection")
        if mongo_query.startswith('db.'):
            parts = mongo_query.split('.')
            if len(parts) >= 2:
                collection = parts[1]
                schemas[db_id].add(collection)
    
    # Convert sets to lists
    schemas = {db: list(collections) for db, collections in schemas.items()}
    
    logger.info(f"Extracted {len(schemas)} database schemas")
    return schemas


def run_pipeline_evaluation(
    config_path: str,
    test_file: str,
    num_samples: int,
    output_dir: str,
    test_start_index: int = 200  # Start testing from index 200 to avoid training samples
) -> Dict[str, Any]:
    """
    Run SMART pipeline evaluation on Spider dev subset
    
    Args:
        config_path: Path to config file
        test_file: Path to spider_dev_mongo.json
        num_samples: Number of samples to evaluate
        output_dir: Output directory for results
        test_start_index: Starting index for test samples (default 200 to avoid training data)
    
    Returns:
        Evaluation results
    """
    logger.info("=" * 80)
    logger.info("SMART Pipeline Evaluation - Spider Dev MongoDB Dataset")
    logger.info("=" * 80)
    
    # Load data with random sampling for test
    import random
    all_data = load_spider_data(test_file, max_samples=None)
    
    # Get random indices from range [test_start_index, len(all_data))
    available_indices = list(range(test_start_index, len(all_data)))
    random.seed(42)  # For reproducibility
    test_indices = random.sample(available_indices, min(num_samples, len(available_indices)))
    test_indices.sort()  # Sort for easier debugging
    
    test_data = [all_data[i] for i in test_indices]
    logger.info(f"Testing on {len(test_data)} random samples from indices {test_start_index}+")
    logger.info(f"Sample indices: {test_indices[:10]}{'...' if len(test_indices) > 10 else ''}")
    
    # Extract schemas
    schemas = extract_schemas_from_data(test_data)
    
    # Create framework
    logger.info("Initializing SMART framework...")
    framework = create_smart_framework(config_path)
    
    # Index training examples for RAG (use separate non-overlapping samples)
    # Strategy: Train on samples 0-99, test on samples 200+
    # This ensures no train/test contamination
    all_data = load_spider_data(test_file, max_samples=None)
    
    # Use first 100 samples for training (plenty of examples, diverse databases)
    training_start = 0
    training_end = min(100, len(all_data))
    training_data = all_data[training_start:training_end]
    
    logger.info(f"Indexing {len(training_data)} training examples for RAG...")
    num_indexed = framework.index_training_examples(training_data)
    logger.info(f"RAG indexing complete: {num_indexed} examples indexed")
    
    logger.info(f"Train samples: {training_start}-{training_end-1}")
    logger.info(f"Test samples: Starting from index {len(test_data)} in full dataset")

    
    # Note: We won't connect to MongoDB for this evaluation since we don't have the actual data
    # We'll just evaluate query generation, not execution
    
    # Run predictions
    predictions = []
    successful = 0
    failed = 0
    
    logger.info(f"\nRunning pipeline on {len(test_data)} samples...")
    logger.info("=" * 80)
    
    for i, item in enumerate(test_data, 1):
        question = item['question']
        gold_query = item['mongo_query']
        db_id = item['db_id']
        
        logger.info(f"\n[{i}/{len(test_data)}] Database: {db_id}")
        logger.info(f"Question: {question}")
        logger.info(f"Gold Query: {gold_query}")
        
        try:
            # Set schema for this database
            # Convert db_id's collection list to collection -> fields mapping
            if db_id in schemas:
                # schemas[db_id] is a list of collection names
                # Convert to {collection_name: []} format expected by schema predictor
                collection_schemas = {coll: [] for coll in schemas[db_id]}
                framework.load_schemas(collection_schemas)
            
            # Translate
            result = framework.translate(question)
            predicted_query = result.get('final_query', '')
            
            logger.info(f"Predicted Query: {predicted_query}")
            logger.info(f"Success: {result.get('success', False)}")
            
            predictions.append({
                'id': item.get('id'),
                'db_id': db_id,
                'question': question,
                'predicted': predicted_query,
                'gold': gold_query,
                'pipeline_success': result.get('success', False)
            })
            
            if result.get('success'):
                successful += 1
            else:
                failed += 1
                
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            predictions.append({
                'id': item.get('id'),
                'db_id': db_id,
                'question': question,
                'predicted': '',
                'gold': gold_query,
                'pipeline_success': False,
                'error': str(e)
            })
            failed += 1
    
    logger.info("\n" + "=" * 80)
    logger.info(f"Pipeline execution: {successful} successful, {failed} failed")
    logger.info("=" * 80)
    
    # Evaluate predictions
    logger.info("\nComputing metrics...")
    evaluator = Evaluator()  # No MongoDB client for query-only evaluation
    
    # Evaluate without execution metrics (since we don't have MongoDB data)
    eval_results = evaluator.evaluate_batch(predictions, compute_execution=False)
    
    # Add metadata
    eval_results['metadata'] = {
        'test_file': test_file,
        'num_samples': num_samples,
        'timestamp': datetime.now().isoformat(),
        'pipeline_successful': successful,
        'pipeline_failed': failed
    }
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    results_file = os.path.join(output_dir, f"spider_eval_pipeline_{num_samples}samples.json")
    with open(results_file, 'w') as f:
        json.dump({
            'evaluation_metrics': eval_results,
            'predictions': predictions
        }, f, indent=2)
    
    logger.info(f"\nResults saved to {results_file}")
    
    # Print summary
    print_evaluation_summary(eval_results)
    
    return eval_results


def print_evaluation_summary(results: Dict[str, Any]):
    """Print evaluation summary"""
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    
    agg = results.get('aggregate', {})
    
    logger.info(f"\nTotal Queries: {results.get('total_queries', 0)}")
    logger.info(f"\nMetrics:")
    logger.info(f"  Exact Match:       {agg.get('exact_match', 0):.2%}")
    logger.info(f"  Collection Match:  {agg.get('collection_match', 0):.2%}")
    logger.info(f"  Operation Match:   {agg.get('operation_match', 0):.2%}")
    
    if 'valid_execution' in agg:
        logger.info(f"  Valid Execution:   {agg.get('valid_execution', 0):.2%}")
    
    logger.info("\n" + "=" * 80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Evaluate SMART Pipeline on Spider Dev MongoDB Dataset"
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--test-file",
        default="data/samples/spider_dev_mongo.json",
        help="Path to spider_dev_mongo.json"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples to evaluate (default: 50)"
    )
    parser.add_argument(
        "--output-dir",
        default="results/spider_eval",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    try:
        # Run evaluation
        run_pipeline_evaluation(
            args.config,
            args.test_file,
            args.num_samples,
            args.output_dir
        )
        
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
