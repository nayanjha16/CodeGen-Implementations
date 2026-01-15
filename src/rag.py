from typing import List, Optional
from sentence_transformers import SentenceTransformer, util
import torch

# Global retriever model
retriever_model: Optional[SentenceTransformer] = None
schema_corpus: List[str] = []  # List of schema descriptions
schema_ids: List[str] = []     # List of corresponding DB IDs
schema_embeddings: Optional[torch.Tensor] = None

def initialize_retriever(schemas: List[dict]) -> None:
    """
    Initializes the retriever with a list of schema dictionaries.
    
    Args:
        schemas: List of dicts with 'text' and 'id' keys
    """
    global retriever_model, schema_corpus, schema_ids, schema_embeddings
    print("Initializing retriever...")
    retriever_model = SentenceTransformer('all-MiniLM-L6-v2')
    
    schema_corpus = [s['text'] for s in schemas]
    schema_ids = [s['id'] for s in schemas]
    
    schema_embeddings = retriever_model.encode(schema_corpus, convert_to_tensor=True)

def retrieve_schema(query: str, top_k: int = 1) -> tuple:
    """
    Retrieves the most relevant schema and its ID.
    
    Returns:
        (schema_text, db_id)
    """
    global retriever_model, schema_embeddings, schema_ids
    
    if retriever_model is None:
        return "", None
        
    query_embedding = retriever_model.encode(query, convert_to_tensor=True)
    hits = util.semantic_search(query_embedding, schema_embeddings, top_k=top_k)
    
    # Just return top 1 for now
    if not hits[0]:
        return "", None
        
    hit = hits[0][0] # Top 1
    idx = hit['corpus_id']
    
    return schema_corpus[idx], schema_ids[idx]
