"""
Execution Optimizer Module
Optimizes queries based on execution results and errors
"""
from typing import Dict, Optional
import logging
from utils.llm_client import BaseLLMClient, PromptTemplate
from utils.mongo_client import MongoDBClient

logger = logging.getLogger(__name__)


class ExecutionOptimizer:
    """Optimizes queries based on execution feedback"""
    
    def __init__(self, llm_client: BaseLLMClient, mongo_client: MongoDBClient):
        """
        Initialize execution optimizer
        
        Args:
            llm_client: LLM client for debugging
            mongo_client: MongoDB client for execution
        """
        self.llm_client = llm_client
        self.mongo_client = mongo_client
        self.prompt_template = PromptTemplate()
        self.max_retries = 5
    
    def optimize(self, nlq: str, query: str, 
                schemas: Dict[str, any]) -> Dict[str, any]:
        """
        Optimize query through execution and debugging
        
        Args:
            nlq: Natural language query
            query: MongoDB query to optimize
            schemas: Database schemas
            
        Returns:
            Dict with 'query', 'success', 'results', and optional 'error'
        """
        current_query = query
        
        for attempt in range(self.max_retries):
            logger.info(f"Execution attempt {attempt + 1}/{self.max_retries}")
            
            # Execute query
            result = self.mongo_client.execute_query(current_query)
            
            if result["success"]:
                logger.info("Query executed successfully")
                return {
                    "query": current_query,
                    "success": True,
                    "results": result["results"],
                    "attempts": attempt + 1
                }
            else:
                # Query failed, try to debug and fix
                logger.warning(f"Query failed: {result.get('error')}")
                
                if attempt < self.max_retries - 1:
                    # Try to fix the query
                    current_query = self._debug_query(
                        nlq=nlq,
                        query=current_query,
                        error=result.get("error", "Unknown error"),
                        schemas=schemas
                    )
                    
                    if not current_query:
                        # Couldn't generate a fix
                        break
                else:
                    # Max retries reached
                    logger.error("Max retries reached, query still failing")
        
        # Return the last execution result (failed)
        return {
            "query": current_query,
            "success": False,
            "error": result.get("error", "Unknown error"),
            "attempts": self.max_retries
        }
    
    def _debug_query(self, nlq: str, query: str, error: str, 
                    schemas: Dict[str, any]) -> str:
        """
        Debug and fix a failing query
        
        Args:
            nlq: Natural language query
            query: Failed query
            error: Error message
            schemas: Database schemas
            
        Returns:
            Corrected query or empty string if unable to fix
        """
        try:
            # Format schemas
            schemas_text = self._format_schemas(schemas)
            
            # Create debugging prompt
            prompts = self.prompt_template.query_debugging_prompt(
                nlq=nlq,
                query=query,
                error=error,
                schemas=schemas_text
            )
            
            # Get corrected query from LLM
            corrected_query = self.llm_client.generate(
                prompt=prompts["user"],
                system_prompt=prompts["system"],
                temperature=0.0,
                max_tokens=1000
            )
            
            # Clean query
            corrected_query = self._clean_query(corrected_query)
            
            logger.info(f"Generated corrected query: {corrected_query}")
            return corrected_query
            
        except Exception as e:
            logger.error(f"Query debugging failed: {e}")
            return ""
    
    def _format_schemas(self, schemas: Dict[str, any]) -> str:
        """Format schemas for prompt"""
        lines = []
        for collection, fields in schemas.items():
            if isinstance(fields, list):
                lines.append(f"Collection: {collection}")
                lines.append(f"Fields: {', '.join(fields)}")
                lines.append("")
        
        return "\n".join(lines)
    
    def _clean_query(self, query: str) -> str:
        """Clean generated query"""
        # Remove markdown code blocks
        if "```" in query:
            parts = query.split("```")
            if len(parts) >= 2:
                query = parts[1]
                if "\n" in query:
                    lines = query.split("\n")
                    if lines[0].strip().lower() in ["javascript", "js", "mongodb", "mongo"]:
                        query = "\n".join(lines[1:])
                    else:
                        query = "\n".join(lines)
        
        query = query.strip()
        
        if query and not query.endswith(";"):
            query += ";"
        
        if "db." in query:
            start_idx = query.find("db.")
            query = query[start_idx:]
            
            first_semicolon = query.find(";")
            if first_semicolon != -1:
                query = query[:first_semicolon + 1]
        
        return query
    
    def validate_query_syntax(self, query: str) -> Dict[str, any]:
        """
        Validate query syntax without execution
        
        Args:
            query: MongoDB query string
            
        Returns:
            Dict with 'valid' bool and optional 'issues' list
        """
        issues = []
        
        # Check if query starts with db.
        if not query.strip().startswith("db."):
            issues.append("Query must start with 'db.'")
        
        # Check if query has a collection name
        if "db." in query:
            parts = query.split(".")
            if len(parts) < 3:
                issues.append("Invalid collection specification")
        
        # Check for balanced braces
        if query.count("{") != query.count("}"):
            issues.append("Unbalanced curly braces")
        
        if query.count("[") != query.count("]"):
            issues.append("Unbalanced square brackets")
        
        if query.count("(") != query.count(")"):
            issues.append("Unbalanced parentheses")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues
        }
