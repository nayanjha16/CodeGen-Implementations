"""
RAG Refiner Module
Refines generated queries using Retrieved Augmented Generation
"""
from typing import Dict, List, Optional
import logging
from utils.llm_client import BaseLLMClient, PromptTemplate
from utils.embeddings import RAGRetriever

logger = logging.getLogger(__name__)


class RAGRefiner:
    """Refines queries using RAG with similar examples"""
    
    def __init__(self, llm_client: BaseLLMClient, rag_retriever: RAGRetriever):
        """
        Initialize RAG refiner
        
        Args:
            llm_client: LLM client for refinement
            rag_retriever: RAG retriever for finding similar examples
        """
        self.llm_client = llm_client
        self.rag_retriever = rag_retriever
        self.prompt_template = PromptTemplate()
    
    def refine(self, nlq: str, initial_query: str, 
               predicted_schema: Optional[Dict] = None,
               num_examples: int = 3) -> str:
        """
        Refine generated query using RAG
        
        Args:
            nlq: Natural language query
            initial_query: Initially generated MongoDB query
            predicted_schema: Optional predicted schema
            num_examples: Number of similar examples to retrieve
            
        Returns:
            Refined MongoDB query
        """
        try:
            # Format predicted schema
            schema_text = ""
            if predicted_schema:
                fields = predicted_schema.get("fields", [])
                collection = predicted_schema.get("collection", "")
                schema_text = f"Collection: {collection}\nFields: {', '.join(fields)}"
            
            # Retrieve similar examples
            similar_examples = self.rag_retriever.retrieve(
                nlq=nlq,
                query=initial_query,
                schema=schema_text,
                top_k=num_examples
            )
            
            if not similar_examples:
                logger.warning("No similar examples found, returning initial query")
                return initial_query
            
            # Create refinement prompt
            prompts = self.prompt_template.query_refinement_prompt(
                nlq=nlq,
                initial_query=initial_query,
                predicted_schemas=schema_text,
                examples=similar_examples
            )
            
            # Get refined query from LLM
            refined_query = self.llm_client.generate(
                prompt=prompts["user"],
                system_prompt=prompts["system"],
                temperature=0.0,
                max_tokens=1000
            )
            
            # Clean query
            refined_query = self._clean_query(refined_query)
            
            logger.info(f"Refined query: {refined_query}")
            logger.info(f"Used {len(similar_examples)} similar examples")
            
            return refined_query
            
        except Exception as e:
            logger.error(f"Query refinement failed: {e}")
            # Return initial query if refinement fails
            return initial_query
    
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
        
        # Extract db.* query if there's explanatory text
        if "db." in query:
            start_idx = query.find("db.")
            query = query[start_idx:]
            
            first_semicolon = query.find(";")
            if first_semicolon != -1:
                query = query[:first_semicolon + 1]
        
        return query
    
    def enable_refinement(self, enable: bool):
        """
        Enable or disable query refinement
        
        Args:
            enable: Whether to enable refinement
        """
        self.enabled = enable
        logger.info(f"RAG refinement {'enabled' if enable else 'disabled'}")
