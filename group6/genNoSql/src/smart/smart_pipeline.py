"""
SMART Pipeline
SLM-assisted and RAG-assisted Multi-step framework for Text-to-NoSQL
"""
from typing import Dict, List, Optional, Any
import logging
from .schema_predictor import SchemaPredictor
from .query_generator import QueryGenerator
from .rag_refiner import RAGRefiner
from .execution_optimizer import ExecutionOptimizer
from utils.mongo_client import MongoDBClient
from utils.llm_client import BaseLLMClient, LLMClientFactory
from utils.embeddings import EmbeddingModel, RAGRetriever

logger = logging.getLogger(__name__)


class SMARTFramework:
    """
    SMART Framework: SLM-assisted and RAG-assisted Multi-step framework
    
    Pipeline:
    1. Schema Prediction: Predict relevant collections and fields
    2. Query Generation: Generate initial MongoDB query
    3. RAG Refinement: Refine query using similar examples
    4. Execution Optimization: Execute and debug query
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SMART framework
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize LLM client
        llm_config = config.get("models", {})
        self.llm_client = LLMClientFactory.create_client(
            provider=llm_config.get("llm_provider", "openai"),
            model=llm_config.get("llm_model", "gpt-4")
        )
        
        # Initialize MongoDB client
        mongo_config = config.get("mongodb", {})
        self.mongo_client = MongoDBClient(
            uri=mongo_config.get("uri", "mongodb://localhost:27017/"),
            database_name=mongo_config.get("database_name", "tend_db"),
            timeout_ms=mongo_config.get("timeout_ms", 5000)
        )
        
        # Initialize components
        self.schema_predictor = SchemaPredictor(self.llm_client)
        self.query_generator = QueryGenerator(self.llm_client)
        
        # Initialize RAG (if enabled)
        rag_config = config.get("rag", {})
        self.rag_enabled = rag_config.get("enabled", True)
        
        if self.rag_enabled:
            # Initialize embedding model and retriever
            embedding_model = EmbeddingModel(
                model_name=rag_config.get("embedding_model", 
                                         "sentence-transformers/all-MiniLM-L6-v2")
            )
            
            self.rag_retriever = RAGRetriever(
                embedding_model=embedding_model,
                nlq_weight=rag_config.get("nlq_weight", 0.4),
                query_weight=rag_config.get("query_weight", 0.3),
                schema_weight=rag_config.get("schema_weight", 0.3)
            )
            
            self.rag_refiner = RAGRefiner(self.llm_client, self.rag_retriever)
        else:
            self.rag_retriever = None
            self.rag_refiner = None
        
        # Initialize execution optimizer
        self.execution_optimizer = ExecutionOptimizer(
            self.llm_client,
            self.mongo_client
        )
        
        # Storage for database schemas
        self.schemas: Dict[str, List[str]] = {}
        
        logger.info("SMART Framework initialized successfully")
    
    def connect_mongodb(self) -> bool:
        """
        Connect to MongoDB
        
        Returns:
            bool: True if successful, False otherwise
        """
        return self.mongo_client.connect()
    
    def load_schemas(self, schemas: Optional[Dict[str, List[str]]] = None):
        """
        Load database schemas
        
        Args:
            schemas: Dict mapping collection names to field lists
                    If None, will introspect from MongoDB
        """
        if schemas:
            self.schemas = schemas
            logger.info(f"Loaded {len(schemas)} collection schemas")
        else:
            # Introspect schemas from MongoDB
            logger.info("Introspecting schemas from MongoDB...")
            collections = self.mongo_client.list_collections()
            
            for collection in collections:
                schema = self.mongo_client.get_collection_schema(collection)
                self.schemas[collection] = schema["fields"]
            
            logger.info(f"Introspected {len(self.schemas)} collection schemas")
    
    def index_training_examples(self, examples: List[Dict]):
        """
        Index training examples for RAG retrieval
        
        Args:
            examples: List of training examples with 'question', 'query' or 'mongo_query', 'schema'
        """
        if not self.rag_enabled or not self.rag_retriever:
            logger.warning("RAG not enabled, skipping indexing")
            return
        
        # Normalize example format - use 'query' key consistently
        normalized_examples = []
        for ex in examples:
            query = ex.get('query', ex.get('mongo_query', ''))
            
            # Normalize to modern MongoDB syntax
            query = query.replace('.count(', '.countDocuments(')
            
            normalized_examples.append({
                'question': ex.get('question', ''),
                'query': query,
                'schema': ex.get('schema', '')
            })
        
        self.rag_retriever.index_examples(normalized_examples)
        logger.info(f"Indexed {len(normalized_examples)} training examples")
    
    def translate(self, nlq: str, use_rag: bool = True, 
                 use_execution_optimization: bool = True) -> Dict[str, Any]:
        """
        Translate natural language query to MongoDB query
        
        Args:
            nlq: Natural language query
            use_rag: Whether to use RAG refinement
            use_execution_optimization: Whether to optimize via execution
            
        Returns:
            Dict with results including final query and intermediate steps
        """
        logger.info(f"Translating NLQ: {nlq}")
        
        result = {
            "nlq": nlq,
            "steps": {}
        }
        
        try:
            # Step 1: Schema Prediction
            logger.info("Step 1: Schema Prediction")
            predicted_schema = self.schema_predictor.predict(nlq, self.schemas)
            result["steps"]["schema_prediction"] = predicted_schema
            
            # Step 2: Query Generation
            logger.info("Step 2: Query Generation")
            initial_query = self.query_generator.generate(
                nlq, self.schemas, predicted_schema
            )
            result["steps"]["initial_query"] = initial_query
            
            # Step 3: RAG Refinement (if enabled)
            if use_rag and self.rag_enabled and self.rag_refiner:
                logger.info("Step 3: RAG Refinement")
                refined_query = self.rag_refiner.refine(
                    nlq=nlq,
                    initial_query=initial_query,
                    predicted_schema=predicted_schema,
                    num_examples=self.config.get("rag", {}).get("num_examples", 3)
                )
                result["steps"]["refined_query"] = refined_query
            else:
                refined_query = initial_query
                result["steps"]["refined_query"] = refined_query
                logger.info("Step 3: RAG Refinement (skipped)")
            
            # Step 4: Execution Optimization (if enabled)
            if use_execution_optimization:
                logger.info("Step 4: Execution Optimization")
                optimization_result = self.execution_optimizer.optimize(
                    nlq=nlq,
                    query=refined_query,
                    schemas=self.schemas
                )
                result["steps"]["optimization"] = optimization_result
                result["final_query"] = optimization_result["query"]
                result["success"] = optimization_result["success"]
                result["results"] = optimization_result.get("results")
                result["error"] = optimization_result.get("error")
            else:
                result["final_query"] = refined_query
                result["success"] = True
                logger.info("Step 4: Execution Optimization (skipped)")
            
            logger.info(f"Translation complete. Final query: {result['final_query']}")
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            result["success"] = False
            result["error"] = str(e)
        
        return result
    
    def batch_translate(self, nlqs: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Translate multiple NLQs
        
        Args:
            nlqs: List of natural language queries
            **kwargs: Additional arguments for translate()
            
        Returns:
            List of translation results
        """
        results = []
        
        for i, nlq in enumerate(nlqs):
            logger.info(f"Processing query {i+1}/{len(nlqs)}")
            result = self.translate(nlq, **kwargs)
            results.append(result)
        
        return results
    
    def disconnect(self):
        """Disconnect from MongoDB"""
        self.mongo_client.disconnect()


def create_smart_framework(config_path: str) -> SMARTFramework:
    """
    Create SMART framework from config file
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Initialized SMARTFramework instance
    """
    import yaml
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    return SMARTFramework(config)
