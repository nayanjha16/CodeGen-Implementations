"""
Main Entry Point for Text-to-NoSQL System
"""
import os
import sys
import argparse
import logging
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from smart.smart_pipeline import create_smart_framework
from evaluation.evaluator import Evaluator
from utils.mongo_client import MongoDBClient
from utils.sample_generator import SampleDataGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_sample_data(mongo_client: MongoDBClient):
    """
    Set up sample data in MongoDB
    
    Args:
        mongo_client: MongoDB client
    """
    logger.info("Setting up sample data...")
    
    # Load sample data
    sample_file = "data/samples/ecommerce.json"
    
    if not os.path.exists(sample_file):
        logger.info("Generating sample data...")
        generator = SampleDataGenerator()
        generator.save_sample_data()
    
    # Load and insert data
    with open(sample_file) as f:
        data = json.load(f)
    
    databases = data["databases"]
    
    for coll_name, coll_data in databases.items():
        logger.info(f"Creating collection: {coll_name}")
        mongo_client.create_collection(coll_name, coll_data["documents"])
    
    logger.info("Sample data setup complete")
    
    return data


def run_demo(config_path: str):
    """
    Run demo of the Text-to-NoSQL system
    
    Args:
        config_path: Path to config file
    """
    logger.info("=" * 60)
    logger.info("Text-to-NoSQL System Demo")
    logger.info("=" * 60)
    
    # Create SMART framework
    framework = create_smart_framework(config_path)
    
    # Connect to MongoDB
    if not framework.connect_mongodb():
        logger.error("Failed to connect to MongoDB")
        logger.info("Please ensure MongoDB is running or update config with MongoDB Atlas URI")
        return
    
    # Setup sample data
    data = setup_sample_data(framework.mongo_client)
    
    # Load schemas
    framework.load_schemas(data["schema_summary"])
    
    # Index training examples for RAG
    training_file = "data/samples/training_examples.json"
    if os.path.exists(training_file):
        with open(training_file) as f:
            examples = json.load(f)
        framework.index_training_examples(examples)
    
    # Demo queries
    demo_queries = [
        "Find all products in the Electronics category",
        "Count the total number of products",
        "Find customers who live in New York",
        "Get the average price of all products",
    ]
    
    logger.info("\n" + "=" * 60)
    logger.info("Running Demo Queries")
    logger.info("=" * 60 + "\n")
    
    for i, nlq in enumerate(demo_queries, 1):
        logger.info(f"\n--- Query {i} ---")
        logger.info(f"Natural Language: {nlq}")
        
        # Translate
        result = framework.translate(nlq)
        
        # Display results
        logger.info(f"Predicted Schema: {result['steps']['schema_prediction']['fields']}")
        logger.info(f"Initial Query: {result['steps']['initial_query']}")
        logger.info(f"Final Query: {result['final_query']}")
        logger.info(f"Success: {result['success']}")
        
        if result.get('results'):
            logger.info(f"Results: {json.dumps(result['results'][:2], indent=2)}...")  # Show first 2 results
        
        if result.get('error'):
            logger.error(f"Error: {result['error']}")
    
    # Disconnect
    framework.disconnect()
    
    logger.info("\n" + "=" * 60)
    logger.info("Demo Complete!")
    logger.info("=" * 60)


def run_evaluation(config_path: str, test_file: str, output_dir: str):
    """
    Run evaluation on test set
    
    Args:
        config_path: Path to config file
        test_file: Path to test queries JSON
        output_dir: Output directory for results
    """
    logger.info("Running evaluation...")
    
    # Create framework
    framework = create_smart_framework(config_path)
    
    # Connect to MongoDB
    if not framework.connect_mongodb():
        logger.error("Failed to connect to MongoDB")
        return
    
    # Load test data
    with open(test_file) as f:
        test_data = json.load(f)
    
    # Load schemas if provided
    if "schema_summary" in test_data:
        framework.load_schemas(test_data["schema_summary"])
    else:
        framework.load_schemas()
    
    # Index training examples
    if "training_examples" in test_data:
        framework.index_training_examples(test_data["training_examples"])
    
    # Run predictions
    test_queries = test_data.get("queries", [])
    predictions = []
    
    for query_item in test_queries:
        nlq = query_item["question"]
        gold_query = query_item["query"]
        
        result = framework.translate(nlq)
        
        predictions.append({
            "question": nlq,
            "predicted": result["final_query"],
            "gold": gold_query,
            "success": result["success"]
        })
    
    # Evaluate
    evaluator = Evaluator(framework.mongo_client)
    eval_results = evaluator.evaluate_batch(predictions, compute_execution=True)
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "evaluation_results.json")
    evaluator.save_results(eval_results, output_file)
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Results")
    logger.info("=" * 60)
    logger.info(f"Total Queries: {eval_results['total_queries']}")
    logger.info(f"Exact Match: {eval_results['aggregate']['exact_match']:.2%}")
    logger.info(f"Collection Match: {eval_results['aggregate']['collection_match']:.2%}")
    logger.info(f"Operation Match: {eval_results['aggregate']['operation_match']:.2%}")
    
    if "execution_accuracy" in eval_results['aggregate']:
        logger.info(f"Execution Accuracy: {eval_results['aggregate']['execution_accuracy']:.2%}")
    
    logger.info(f"\nResults saved to: {output_file}")
    
    # Disconnect
    framework.disconnect()


def interactive_mode(config_path: str):
    """
    Interactive mode for querying
    
    Args:
        config_path: Path to config file
    """
    logger.info("Starting interactive mode...")
    logger.info("Type 'quit' or 'exit' to stop\n")
    
    # Create framework
    framework = create_smart_framework(config_path)
    
    # Connect to MongoDB
    if not framework.connect_mongodb():
        logger.error("Failed to connect to MongoDB")
        return
    
    # Load schemas
    framework.load_schemas()
    
    while True:
        try:
            nlq = input("\nEnter natural language query: ").strip()
            
            if nlq.lower() in ['quit', 'exit']:
                break
            
            if not nlq:
                continue
            
            # Translate
            result = framework.translate(nlq)
            
            # Display
            print(f"\nGenerated Query: {result['final_query']}")
            print(f"Success: {result['success']}")
            
            if result.get('results'):
                print(f"\nResults ({len(result['results'])} documents):")
                print(json.dumps(result['results'][:3], indent=2))
            
            if result.get('error'):
                print(f"\nError: {result['error']}")
        
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error: {e}")
    
    # Disconnect
    framework.disconnect()
    logger.info("\nGoodbye!")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Text-to-NoSQL System")
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--mode",
        choices=["demo", "eval", "interactive"],
        default="demo",
        help="Run mode"
    )
    parser.add_argument(
        "--test-file",
        help="Test file for evaluation mode"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Output directory for evaluation results"
    )
    
    args = parser.parse_args()
    
    try:
        if args.mode == "demo":
            run_demo(args.config)
        elif args.mode == "eval":
            if not args.test_file:
                logger.error("--test-file required for evaluation mode")
                return
            run_evaluation(args.config, args.test_file, args.output_dir)
        elif args.mode == "interactive":
            interactive_mode(args.config)
    
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
