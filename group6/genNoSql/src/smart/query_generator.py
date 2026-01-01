"""
Query Generator Module
Generates MongoDB queries from natural language
"""
from typing import Dict, List, Optional
import logging
from utils.llm_client import BaseLLMClient, PromptTemplate

logger = logging.getLogger(__name__)


class QueryGenerator:
    """Generates MongoDB queries from natural language"""
    
    def __init__(self, llm_client: BaseLLMClient):
        """
        Initialize query generator
        
        Args:
            llm_client: LLM client for generation
        """
        self.llm_client = llm_client
        self.prompt_template = PromptTemplate()
    
    def generate(self, nlq: str, schemas: Dict[str, List[str]], 
                 predicted_schema: Optional[Dict] = None) -> str:
        """
        Generate MongoDB query from NLQ
        
        Args:
            nlq: Natural language query
            schemas: Dict mapping collection names to field lists
            predicted_schema: Optional predicted schema from SchemaPredictor
            
        Returns:
            MongoDB query string
        """
        # Format schemas
        schemas_text = self._format_schemas(schemas)
        
        # Add predicted schema hint if available
        if predicted_schema and predicted_schema.get("collection"):
            collection = predicted_schema["collection"]
            schemas_text = f"**Focus on Collection: {collection}**\n\n{schemas_text}"
        
        # Create prompt
        prompts = self.prompt_template.query_generation_prompt(nlq, schemas_text)
        
        try:
            # Generate query
            query = self.llm_client.generate(
                prompt=prompts["user"],
                system_prompt=prompts["system"],
                temperature=0.0,
                max_tokens=1000
            )
            
            # Clean and validate query
            query = self._clean_query(query)
            
            logger.info(f"Generated query: {query}")
            return query
            
        except Exception as e:
            logger.error(f"Query generation failed: {e}")
            return ""
    
    def _format_schemas(self, schemas: Dict[str, List[str]]) -> str:
        """Format schemas for prompt"""
        lines = []
        for collection, fields in schemas.items():
            lines.append(f"Collection: {collection}")
            lines.append(f"Fields: {', '.join(fields)}")
            lines.append("")
        
        return "\n".join(lines)
    
    def _clean_query(self, query: str) -> str:
        """
        Clean and validate generated query
        
        Args:
            query: Raw query from LLM
            
        Returns:
            Cleaned query string
        """
        # Remove markdown code blocks
        if "```" in query:
            parts = query.split("```")
            if len(parts) >= 2:
                query = parts[1]
                # Remove language identifier (javascript, js, mongodb, etc.)
                if "\n" in query:
                    lines = query.split("\n")
                    if lines[0].strip().lower() in ["javascript", "js", "mongodb", "mongo"]:
                        query = "\n".join(lines[1:])
                    else:
                        query = "\n".join(lines)
        
        # Trim whitespace
        query = query.strip()
        
        # Ensure query ends with semicolon
        if query and not query.endswith(";"):
            query += ";"
        
        # Remove any explanatory text before/after the query
        # Query should start with "db."
        if "db." in query:
            # Find the first occurrence of "db."
            start_idx = query.find("db.")
            query = query[start_idx:]
            
            # If there are multiple queries, take the first one
            if query.count("db.") > 1:
                # Find the end of the first query (semicolon)
                first_semicolon = query.find(";")
                if first_semicolon != -1:
                    query = query[:first_semicolon + 1]
        
        return query
    
    def batch_generate(self, queries: List[Dict[str, any]], 
                      schemas: Dict[str, List[str]]) -> List[str]:
        """
        Generate queries for multiple NLQs
        
        Args:
            queries: List of dicts with 'question' key
            schemas: MongoDB schemas
            
        Returns:
            List of generated queries
        """
        results = []
        
        for item in queries:
            nlq = item.get("question", "")
            predicted_schema = item.get("predicted_schema")
            
            query = self.generate(nlq, schemas, predicted_schema)
            results.append(query)
        
        return results


class FineTunedQueryGenerator(QueryGenerator):
    """Query generator using fine-tuned SLM"""
    
    def __init__(self, model_path: str):
        """
        Initialize with fine-tuned model
        
        Args:
            model_path: Path to fine-tuned model
        """
        from utils.llm_client import LocalLLMClient
        
        llm_client = LocalLLMClient(model_path)
        super().__init__(llm_client)
        
        logger.info(f"Loaded fine-tuned query generator from {model_path}")
