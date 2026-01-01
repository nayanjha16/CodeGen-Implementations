"""
Embeddings Utility
Handles text embeddings for RAG similarity search
"""
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Wrapper for embedding models"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize embedding model
        
        Args:
            model_name: Name of the sentence-transformers model
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
        
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully")
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts to embeddings
        
        Args:
            texts: List of text strings
            
        Returns:
            numpy array of embeddings
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings
    
    def encode_single(self, text: str) -> np.ndarray:
        """
        Encode a single text to embedding
        
        Args:
            text: Text string
            
        Returns:
            numpy array embedding
        """
        return self.encode([text])[0]


class RAGRetriever:
    """Retrieval-Augmented Generation retriever for finding similar examples"""
    
    def __init__(self, embedding_model: EmbeddingModel,
                 nlq_weight: float = 0.4,
                 query_weight: float = 0.3,
                 schema_weight: float = 0.3):
        """
        Initialize RAG retriever
        
        Args:
            embedding_model: Embedding model instance
            nlq_weight: Weight for NLQ similarity
            query_weight: Weight for query similarity
            schema_weight: Weight for schema similarity
        """
        self.embedding_model = embedding_model
        self.nlq_weight = nlq_weight
        self.query_weight = query_weight
        self.schema_weight = schema_weight
        
        # Validate weights
        total_weight = nlq_weight + query_weight + schema_weight
        if not np.isclose(total_weight, 1.0):
            logger.warning(
                f"Weights sum to {total_weight}, normalizing to 1.0"
            )
            self.nlq_weight /= total_weight
            self.query_weight /= total_weight
            self.schema_weight /= total_weight
        
        # Storage for indexed examples
        self.examples: List[Dict] = []
        self.nlq_embeddings: np.ndarray = None
        self.query_embeddings: np.ndarray = None
        self.schema_embeddings: np.ndarray = None
    
    def index_examples(self, examples: List[Dict]):
        """
        Index training examples for retrieval
        
        Args:
            examples: List of dicts with 'question', 'query', 'schema' keys
        """
        logger.info(f"Indexing {len(examples)} examples...")
        
        self.examples = examples
        
        # Extract texts
        nlq_texts = [ex.get('question', '') for ex in examples]
        query_texts = [ex.get('query', '') for ex in examples]
        schema_texts = [ex.get('schema', '') for ex in examples]
        
        # Generate embeddings
        self.nlq_embeddings = self.embedding_model.encode(nlq_texts)
        self.query_embeddings = self.embedding_model.encode(query_texts)
        self.schema_embeddings = self.embedding_model.encode(schema_texts)
        
        logger.info("Indexing complete")
    
    def retrieve(self, nlq: str, query: str = "", schema: str = "",
                 top_k: int = 3) -> List[Dict]:
        """
        Retrieve top-k similar examples
        
        Args:
            nlq: Natural language query
            query: Generated MongoDB query (optional)
            schema: Predicted schema (optional)
            top_k: Number of examples to retrieve
            
        Returns:
            List of top-k most similar examples
        """
        if not self.examples:
            logger.warning("No examples indexed, returning empty list")
            return []
        
        # Encode query components
        nlq_emb = self.embedding_model.encode_single(nlq)
        query_emb = self.embedding_model.encode_single(query) if query else None
        schema_emb = self.embedding_model.encode_single(schema) if schema else None
        
        # Calculate similarities
        nlq_sim = cosine_similarity([nlq_emb], self.nlq_embeddings)[0]
        
        # Calculate weighted similarity
        if query_emb is not None and schema_emb is not None:
            query_sim = cosine_similarity([query_emb], self.query_embeddings)[0]
            schema_sim = cosine_similarity([schema_emb], self.schema_embeddings)[0]
            
            combined_sim = (
                self.nlq_weight * nlq_sim +
                self.query_weight * query_sim +
                self.schema_weight * schema_sim
            )
        else:
            # If query/schema not provided, use only NLQ similarity
            combined_sim = nlq_sim
        
        # Get top-k indices
        top_indices = np.argsort(combined_sim)[-top_k:][::-1]
        
        # Return top-k examples with similarity scores
        results = []
        for idx in top_indices:
            example = self.examples[idx].copy()
            example['similarity'] = float(combined_sim[idx])
            results.append(example)
        
        return results
    
    def format_schemas(self, schema_dict: Dict[str, List[str]]) -> str:
        """
        Format schema dictionary to string representation
        
        Args:
            schema_dict: Dict mapping collection names to field lists
            
        Returns:
            Formatted schema string
        """
        lines = []
        for collection, fields in schema_dict.items():
            lines.append(f"Collection: {collection}")
            lines.append(f"Fields: {', '.join(fields)}")
            lines.append("")
        
        return "\n".join(lines)


def compute_similarity(text1: str, text2: str, 
                      embedding_model: EmbeddingModel) -> float:
    """
    Compute cosine similarity between two texts
    
    Args:
        text1: First text
        text2: Second text
        embedding_model: Embedding model to use
        
    Returns:
        Similarity score (0-1)
    """
    emb1 = embedding_model.encode_single(text1)
    emb2 = embedding_model.encode_single(text2)
    
    similarity = cosine_similarity([emb1], [emb2])[0][0]
    return float(similarity)
