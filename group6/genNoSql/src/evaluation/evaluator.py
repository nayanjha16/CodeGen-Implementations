"""
Evaluation Module
Metrics and evaluation for Text-to-NoSQL system
"""
from typing import Dict, List, Any, Optional
import logging
import json
from utils.mongo_client import MongoDBClient

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluator for Text-to-NoSQL system"""
    
    def __init__(self, mongo_client: Optional[MongoDBClient] = None):
        """
        Initialize evaluator
        
        Args:
            mongo_client: MongoDB client for execution-based metrics
        """
        self.mongo_client = mongo_client
    
    def exact_match(self, predicted_query: str, gold_query: str) -> bool:
        """
        Exact match metric
        
        Args:
            predicted_query: Predicted query
            gold_query: Gold standard query
            
        Returns:
            bool: True if exact match
        """
        # Normalize queries
        pred_norm = self._normalize_query(predicted_query)
        gold_norm = self._normalize_query(gold_query)
        
        return pred_norm == gold_norm
    
    def query_component_match(self, predicted_query: str, gold_query: str) -> Dict[str, bool]:
        """
        Component-wise query comparison
        
        Args:
            predicted_query: Predicted query
            gold_query: Gold standard query
            
        Returns:
            Dict with component match results
        """
        pred_components = self._extract_components(predicted_query)
        gold_components = self._extract_components(gold_query)
        
        return {
            "collection_match": pred_components["collection"] == gold_components["collection"],
            "operation_match": pred_components["operation"] == gold_components["operation"],
            "filter_match": pred_components.get("filter") == gold_components.get("filter"),
            "projection_match": pred_components.get("projection") == gold_components.get("projection")
        }
    
    def execution_accuracy(self, predicted_query: str, gold_query: str) -> bool:
        """
        Execution accuracy metric
        
        Args:
            predicted_query: Predicted query
            gold_query: Gold standard query
            
        Returns:
            bool: True if both queries return same results
        """
        if not self.mongo_client:
            logger.warning("MongoDB client not provided, cannot compute execution accuracy")
            return False
        
        # Execute both queries
        pred_result = self.mongo_client.execute_query(predicted_query)
        gold_result = self.mongo_client.execute_query(gold_query)
        
        # Check if both succeeded
        if not pred_result["success"] or not gold_result["success"]:
            return False
        
        # Compare results
        return self._compare_results(pred_result["results"], gold_result["results"])
    
    def valid_execution(self, query: str) -> bool:
        """
        Valid execution metric
        
        Args:
            query: MongoDB query
            
        Returns:
            bool: True if query executes without errors
        """
        if not self.mongo_client:
            logger.warning("MongoDB client not provided, cannot check execution")
            return False
        
        result = self.mongo_client.execute_query(query)
        return result["success"]
    
    def evaluate_single(self, predicted_query: str, gold_query: str,
                       compute_execution: bool = True) -> Dict[str, Any]:
        """
        Evaluate a single query prediction
        
        Args:
            predicted_query: Predicted query
            gold_query: Gold standard query
            compute_execution: Whether to compute execution-based metrics
            
        Returns:
            Dict with all metric scores
        """
        metrics = {}
        
        # Query-based metrics
        metrics["exact_match"] = self.exact_match(predicted_query, gold_query)
        metrics["component_match"] = self.query_component_match(predicted_query, gold_query)
        
        # Execution-based metrics
        if compute_execution and self.mongo_client:
            metrics["valid_execution"] = self.valid_execution(predicted_query)
            metrics["execution_accuracy"] = self.execution_accuracy(predicted_query, gold_query)
        
        return metrics
    
    def evaluate_batch(self, predictions: List[Dict[str, str]],
                      compute_execution: bool = True) -> Dict[str, Any]:
        """
        Evaluate multiple predictions
        
        Args:
            predictions: List of dicts with 'predicted' and 'gold' keys
            compute_execution: Whether to compute execution-based metrics
            
        Returns:
            Dict with aggregate metrics
        """
        results = []
        
        for i, pred in enumerate(predictions):
            logger.info(f"Evaluating query {i+1}/{len(predictions)}")
            
            result = self.evaluate_single(
                predicted_query=pred["predicted"],
                gold_query=pred["gold"],
                compute_execution=compute_execution
            )
            
            result["question"] = pred.get("question", "")
            results.append(result)
        
        # Compute aggregate statistics
        aggregate = self._aggregate_metrics(results)
        
        return {
            "individual_results": results,
            "aggregate": aggregate,
            "total_queries": len(predictions)
        }
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query string for comparison"""
        # Remove whitespace, semicolons
        normalized = query.strip().rstrip(";")
        
        # Remove extra whitespace
        import re
        normalized = re.sub(r'\s+', ' ', normalized)
        
        return normalized
    
    def _extract_components(self, query: str) -> Dict[str, str]:
        """Extract query components"""
        import re
        
        components = {
            "collection": None,
            "operation": None,
            "filter": None,
            "projection": None
        }
        
        # Extract collection name
        coll_match = re.search(r'db\.(\w+)', query)
        if coll_match:
            components["collection"] = coll_match.group(1)
        
        # Extract operation (find, aggregate, etc.)
        op_match = re.search(r'\.(\w+)\(', query)
        if op_match:
            components["operation"] = op_match.group(1)
        
        # Extract filter and projection (for find queries)
        if ".find(" in query:
            find_match = re.search(r'\.find\((.*?)\)', query)
            if find_match:
                params = find_match.group(1)
                # This is simplified - full parsing would need proper JSON parser
                components["filter"] = params
        
        return components
    
    def _compare_results(self, results1: List[Dict], results2: List[Dict]) -> bool:
        """Compare query execution results"""
        # Simple comparison - check if same length and same content
        if len(results1) != len(results2):
            return False
        
        # Sort and compare (order may differ)
        try:
            sorted1 = sorted(results1, key=lambda x: json.dumps(x, sort_keys=True))
            sorted2 = sorted(results2, key=lambda x: json.dumps(x, sort_keys=True))
            
            return sorted1 == sorted2
        except:
            # If sorting fails, do direct comparison
            return results1 == results2
    
    def _aggregate_metrics(self, results: List[Dict]) -> Dict[str, float]:
        """Aggregate individual metrics"""
        n = len(results)
        if n == 0:
            return {}
        
        aggregate = {
            "exact_match": sum(r["exact_match"] for r in results) / n,
            "collection_match": sum(
                r["component_match"]["collection_match"] for r in results
            ) / n,
            "operation_match": sum(
                r["component_match"]["operation_match"] for r in results
            ) / n
        }
        
        # Add execution metrics if present
        if "valid_execution" in results[0]:
            aggregate["valid_execution"] = sum(
                r["valid_execution"] for r in results
            ) / n
        
        if "execution_accuracy" in results[0]:
            aggregate["execution_accuracy"] = sum(
                r["execution_accuracy"] for r in results
            ) / n
        
        return aggregate
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """
        Save evaluation results to JSON file
        
        Args:
            results: Evaluation results dict
            output_path: Output file path
        """
        import os
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Saved results to {output_path}")
