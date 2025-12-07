from typing import List, Optional
from sentence_transformers import SentenceTransformer, util
import torch

# Global retriever model (lazy loading recommended in production)
retriever_model: Optional[SentenceTransformer] = None
schema_corpus: List[str] = []  # List of schema descriptions
schema_embeddings: Optional[torch.Tensor] = None

def initialize_retriever(schemas: List[str]) -> None:
    """
    Initializes the retriever with a list of schema strings.
    
    Args:
        schemas: List of database schema descriptions as strings
    """
    global retriever_model, schema_corpus, schema_embeddings
    print("Initializing retriever...")
    retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
    schema_corpus = schemas
    schema_embeddings = retriever_model.encode(schema_corpus, convert_to_tensor=True)

def retrieve_schema(query: str, top_k: int = 1) -> str:
    """
    Retrieves the most relevant schema for the query using semantic search.
    
    Args:
        query: Natural language query from user
        top_k: Number of top schemas to retrieve (default: 1)
        
    Returns:
        Retrieved schema descriptions joined by newlines
    """
    global retriever_model, schema_embeddings
    
    if retriever_model is None:
        # Fallback if not initialized
        return ""
        
    query_embedding = retriever_model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, schema_embeddings, top_k=top_k)
    
    results: List[str] = []
    for hit in hits[0]:
        results.append(schema_corpus[hit['corpus_id']])
        
    return "\n\n".join(results)
