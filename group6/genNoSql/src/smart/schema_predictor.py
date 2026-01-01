"""
Schema Predictor Module
Predicts relevant MongoDB schema elements (collections, fields) from NLQ
"""
from typing import Dict, List, Optional
import logging
from utils.llm_client import BaseLLMClient, PromptTemplate

logger = logging.getLogger(__name__)


class SchemaPredictor:
    """Predicts MongoDB schema elements from natural language query"""
    
    def __init__(self, llm_client: BaseLLMClient):
        """
        Initialize schema predictor
        
        Args:
            llm_client: LLM client for predictions
        """
        self.llm_client = llm_client
        self.prompt_template = PromptTemplate()
    
    def predict(self, nlq: str, schemas: Dict[str, List[str]]) -> Dict[str, any]:
        """
        Predict relevant schema elements from NLQ
        
        Args:
            nlq: Natural language query
            schemas: Dict mapping collection names to field lists
            
        Returns:
            Dict with predicted collections, fields, etc.
        """
        # Format schemas for prompt
        schemas_text = self._format_schemas(schemas)
        
        # Create prompt
        prompts = self.prompt_template.schema_prediction_prompt(nlq, schemas_text)
        
        try:
            # Get prediction from LLM
            response = self.llm_client.generate(
                prompt=prompts["user"],
                system_prompt=prompts["system"],
                temperature=0.0,
                max_tokens=500
            )
            
            # Parse response
            predicted_fields = self._parse_schema_response(response)
            
            # Infer collection from fields
            predicted_collection = self._infer_collection(predicted_fields, schemas)
            
            result = {
                "fields": predicted_fields,
                "collection": predicted_collection,
                "raw_response": response
            }
            
            logger.info(f"Predicted schema: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Schema prediction failed: {e}")
            return {
                "fields": [],
                "collection": None,
                "error": str(e)
            }
    
    def _format_schemas(self, schemas: Dict[str, List[str]]) -> str:
        """
        Format schema dictionary to string
        
        Args:
            schemas: Dict mapping collection names to field lists
            
        Returns:
            Formatted string
        """
        lines = []
        for collection, fields in schemas.items():
            lines.append(f"Collection: {collection}")
            lines.append(f"Fields: {', '.join(fields)}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _parse_schema_response(self, response: str) -> List[str]:
        """
        Parse LLM response to extract field names
        
        Args:
            response: LLM response text
            
        Returns:
            List of field names
        """
        # Clean response
        response = response.strip()
        
        # Remove any markdown code blocks
        if "```" in response:
            # Extract content from code block
            parts = response.split("```")
            if len(parts) >= 2:
                response = parts[1]
                # Remove language identifier
                if "\n" in response:
                    response = response.split("\n", 1)[1]
        
        # Split by comma and clean
        fields = [f.strip() for f in response.split(",")]
        fields = [f for f in fields if f]  # Remove empty strings
        
        return fields
    
    def _infer_collection(self, fields: List[str], 
                         schemas: Dict[str, List[str]]) -> Optional[str]:
        """
        Infer which collection is being queried based on predicted fields
        
        Args:
            fields: List of predicted field names
            schemas: Dict mapping collection names to field lists
            
        Returns:
            Most likely collection name
        """
        if not fields:
            return None
        
        # Count how many predicted fields match each collection
        scores = {}
        for collection, coll_fields in schemas.items():
            # Check both exact matches and nested field matches
            score = 0
            for field in fields:
                # Direct match
                if field in coll_fields:
                    score += 1
                # Nested field match (e.g., "orders.total" matches "orders")
                elif "." in field:
                    base_field = field.split(".")[0]
                    if base_field in coll_fields:
                        score += 0.5
            
            scores[collection] = score
        
        # Return collection with highest score
        if scores:
            best_collection = max(scores.items(), key=lambda x: x[1])
            if best_collection[1] > 0:
                return best_collection[0]
        
        return None


class FineTunedSchemaPredictor(SchemaPredictor):
    """Schema predictor using fine-tuned SLM"""
    
    def __init__(self, model_path: str):
        """
        Initialize with fine-tuned model
        
        Args:
            model_path: Path to fine-tuned model
        """
        from utils.llm_client import LocalLLMClient
        
        llm_client = LocalLLMClient(model_path)
        super().__init__(llm_client)
        
        logger.info(f"Loaded fine-tuned schema predictor from {model_path}")
