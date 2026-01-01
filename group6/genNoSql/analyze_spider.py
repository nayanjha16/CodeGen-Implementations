"""
Comprehensive Analysis Script for Spider Dev MongoDB Dataset
Analyzes existing MongoDB conversions and compares with SMART pipeline results
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_existing_conversions(file_path: str) -> Dict[str, Any]:
    """
    Analyze existing MongoDB conversions in the dataset
    
    Args:
        file_path: Path to spider_dev_mongo.json
    
    Returns:
        Analysis results
    """
    logger.info("=" * 80)
    logger.info("COMPREHENSIVE ANALYSIS: Spider Dev MongoDB Dataset")
    logger.info("=" * 80)
    
    # Load data
    logger.info(f"\nLoading data from {file_path}")
    with open(file_path) as f:
        data = json.load(f)
    
    total_queries = len(data)
    logger.info(f"Total queries: {total_queries}")
    
    # Initialize counters
    successful = 0
    failed = 0
    errors_by_type = defaultdict(int)
    queries_by_db = defaultdict(int)
    success_by_db = defaultdict(int)
    failed_by_db = defaultdict(int)
    
    # Query type analysis
    query_types = {
        'find': 0,
        'aggregate': 0,
        'count': 0,
        'distinct': 0,
        'other': 0
    }
    
    # Error patterns
    error_patterns = {
        'subquery': 0,
        'intersect/except': 0,
        'distinct_multiple_columns': 0,
        'null_pointer': 0,
        'parse_exception': 0,
        'other': 0
    }
    
    # Analyze each query
    for item in data:
        db_id = item.get('db_id', 'unknown')
        success = item.get('success', False)
        mongo_query = item.get('mongo_query', '')
        error = item.get('error', '')
        
        queries_by_db[db_id] += 1
        
        if success:
            successful += 1
            success_by_db[db_id] += 1
            
            # Classify query type
            if mongo_query:
                if '.find(' in mongo_query:
                    query_types['find'] += 1
                elif '.aggregate(' in mongo_query:
                    query_types['aggregate'] += 1
                elif '.count(' in mongo_query:
                    query_types['count'] += 1
                elif '.distinct(' in mongo_query:
                    query_types['distinct'] += 1
                else:
                    query_types['other'] += 1
        else:
            failed += 1
            failed_by_db[db_id] += 1
            
            # Classify error type
            if error:
                error_lower = error.lower()
                
                if 'select' in error_lower and ('subquery' in error_lower or 'parseNaturalLanguageDate' in error):
                    error_patterns['subquery'] += 1
                    errors_by_type['Subquery not supported'] += 1
                elif 'intersect' in error_lower or 'except' in error_lower:
                    error_patterns['intersect/except'] += 1
                    errors_by_type['INTERSECT/EXCEPT not supported'] += 1
                elif 'cannot run distinct' in error_lower and 'more than one column' in error_lower:
                    error_patterns['distinct_multiple_columns'] += 1
                    errors_by_type['DISTINCT on multiple columns'] += 1
                elif 'nullpointerexception' in error_lower:
                    error_patterns['null_pointer'] += 1
                    errors_by_type['NullPointerException'] += 1
                elif 'parseexception' in error_lower:
                    error_patterns['parse_exception'] += 1
                    errors_by_type['Parse exception'] += 1
                else:
                    error_patterns['other'] += 1
                    errors_by_type['Other errors'] += 1
    
    # Calculate statistics
    success_rate = successful / total_queries if total_queries > 0 else 0
    
    # Create results
    results = {
        'summary': {
            'total_queries': total_queries,
            'successful': successful,
            'failed': failed,
            'success_rate': success_rate
        },
        'query_types': dict(query_types),
        'databases': {
            'total_databases': len(queries_by_db),
            'queries_per_db': dict(queries_by_db),
            'success_per_db': dict(success_by_db),
            'failed_per_db': dict(failed_by_db),
            'success_rate_per_db': {
                db: success_by_db[db] / queries_by_db[db]
                for db in queries_by_db
            }
        },
        'error_analysis': {
            'error_patterns': dict(error_patterns),
            'errors_by_type': dict(errors_by_type)
        },
        'metadata': {
            'file': file_path,
            'timestamp': datetime.now().isoformat()
        }
    }
    
    return results


def print_analysis_report(results: Dict[str, Any]):
    """Print comprehensive analysis report"""
    logger.info("\n" + "=" * 80)
    logger.info("ANALYSIS REPORT")
    logger.info("=" * 80)
    
    # Summary
    summary = results['summary']
    logger.info(f"\n📊 OVERALL STATISTICS")
    logger.info(f"{'─' * 80}")
    logger.info(f"Total Queries:     {summary['total_queries']}")
    logger.info(f"Successful:        {summary['successful']} ({summary['success_rate']:.2%})")
    logger.info(f"Failed:            {summary['failed']} ({(1-summary['success_rate']):.2%})")
    
    # Query types
    logger.info(f"\n🔍 QUERY TYPE DISTRIBUTION")
    logger.info(f"{'─' * 80}")
    query_types = results['query_types']
    for qtype, count in sorted(query_types.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            pct = count / summary['successful'] * 100 if summary['successful'] > 0 else 0
            logger.info(f"{qtype.ljust(15)} {count:5d}  ({pct:.1f}%)")
    
    # Database analysis
    logger.info(f"\n💾 DATABASE ANALYSIS")
    logger.info(f"{'─' * 80}")
    db_info = results['databases']
    logger.info(f"Total Databases:   {db_info['total_databases']}")
    
    logger.info(f"\nTop 10 Databases by Query Count:")
    sorted_dbs = sorted(
        db_info['queries_per_db'].items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    for db, count in sorted_dbs:
        success_rate = db_info['success_rate_per_db'].get(db, 0)
        logger.info(f"  {db.ljust(25)} {count:4d} queries  ({success_rate:.1%} success)")
    
    # Error analysis
    logger.info(f"\n❌ ERROR ANALYSIS")
    logger.info(f"{'─' * 80}")
    error_patterns = results['error_analysis']['error_patterns']
    
    logger.info("Error Pattern Distribution:")
    for pattern, count in sorted(error_patterns.items(), key=lambda x: x[1], reverse=True):
        if count > 0:
            pct = count / summary['failed'] * 100 if summary['failed'] > 0 else 0
            logger.info(f"  {pattern.ljust(30)} {count:4d}  ({pct:.1f}%)")
    
    logger.info(f"\nTop Error Types:")
    errors_by_type = results['error_analysis']['errors_by_type']
    for error_type, count in sorted(errors_by_type.items(), key=lambda x: x[1], reverse=True)[:5]:
        logger.info(f"  {error_type.ljust(35)} {count:4d}")
    
    logger.info("\n" + "=" * 80)


def compare_with_pipeline_results(
    baseline_file: str,
    pipeline_results_file: str,
    output_dir: str
) -> Dict[str, Any]:
    """
    Compare baseline conversions with SMART pipeline results
    
    Args:
        baseline_file: Path to spider_dev_mongo.json
        pipeline_results_file: Path to pipeline evaluation results
        output_dir: Output directory
    
    Returns:
        Comparison results
    """
    logger.info("\n" + "=" * 80)
    logger.info("COMPARISON: Baseline vs SMART Pipeline")
    logger.info("=" * 80)
    
    # Load baseline data
    with open(baseline_file) as f:
        baseline_data = json.load(f)
    
    # Load pipeline results
    with open(pipeline_results_file) as f:
        pipeline_data = json.load(f)
    
    # Create lookup for baseline
    baseline_lookup = {item['id']: item for item in baseline_data}
    
    # Compare
    comparisons = []
    matches = 0
    pipeline_better = 0
    baseline_better = 0
    
    for pred in pipeline_data.get('predictions', []):
        item_id = pred.get('id')
        baseline_item = baseline_lookup.get(item_id)
        
        if not baseline_item:
            continue
        
        pipeline_success = pred.get('pipeline_success', False)
        baseline_success = baseline_item.get('success', False)
        
        # Check if queries match
        pred_query = pred.get('predicted', '').strip()
        baseline_query = baseline_item.get('mongo_query', '').strip()
        
        queries_match = pred_query == baseline_query
        
        if queries_match:
            matches += 1
        
        if pipeline_success and not baseline_success:
            pipeline_better += 1
        elif baseline_success and not pipeline_success:
            baseline_better += 1
        
        comparisons.append({
            'id': item_id,
            'question': pred.get('question'),
            'baseline_success': baseline_success,
            'pipeline_success': pipeline_success,
            'queries_match': queries_match,
            'baseline_query': baseline_query,
            'pipeline_query': pred_query
        })
    
    results = {
        'total_compared': len(comparisons),
        'exact_matches': matches,
        'pipeline_better': pipeline_better,
        'baseline_better': baseline_better,
        'both_successful': sum(1 for c in comparisons if c['baseline_success'] and c['pipeline_success']),
        'both_failed': sum(1 for c in comparisons if not c['baseline_success'] and not c['pipeline_success']),
        'comparisons': comparisons
    }
    
    # Save comparison
    os.makedirs(output_dir, exist_ok=True)
    comparison_file = os.path.join(output_dir, 'baseline_vs_pipeline_comparison.json')
    with open(comparison_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nComparison saved to {comparison_file}")
    
    # Print summary
    logger.info(f"\n📈 COMPARISON SUMMARY")
    logger.info(f"{'─' * 80}")
    logger.info(f"Total Compared:        {results['total_compared']}")
    logger.info(f"Exact Matches:         {results['exact_matches']} ({results['exact_matches']/results['total_compared']:.1%})")
    logger.info(f"Pipeline Better:       {results['pipeline_better']}")
    logger.info(f"Baseline Better:       {results['baseline_better']}")
    logger.info(f"Both Successful:       {results['both_successful']}")
    logger.info(f"Both Failed:           {results['both_failed']}")
    
    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Comprehensive Analysis of Spider Dev MongoDB Dataset"
    )
    parser.add_argument(
        "--baseline-file",
        default="data/samples/spider_dev_mongo.json",
        help="Path to spider_dev_mongo.json"
    )
    parser.add_argument(
        "--pipeline-results",
        help="Path to pipeline evaluation results (optional)"
    )
    parser.add_argument(
        "--output-dir",
        default="results/spider_analysis",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    try:
        # Analyze existing conversions
        logger.info("Starting comprehensive analysis...\n")
        analysis_results = analyze_existing_conversions(args.baseline_file)
        
        # Print report
        print_analysis_report(analysis_results)
        
        # Save analysis
        os.makedirs(args.output_dir, exist_ok=True)
        analysis_file = os.path.join(args.output_dir, 'baseline_analysis.json')
        with open(analysis_file, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        logger.info(f"\nAnalysis saved to {analysis_file}")
        
        # Compare with pipeline if results provided
        if args.pipeline_results and os.path.exists(args.pipeline_results):
            logger.info("\nPipeline results found, running comparison...")
            compare_with_pipeline_results(
                args.baseline_file,
                args.pipeline_results,
                args.output_dir
            )
        
        return 0
        
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
