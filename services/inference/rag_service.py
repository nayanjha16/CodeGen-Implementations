from sentence_transformers import SentenceTransformer
import json
import numpy as np
from typing import List, Dict

class RAGService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the RAG service with a sentence-transformer model.
        """
        print(f"Loading RAG model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.schema_embeddings = {} # Cache for schema embeddings: {db_id: {table_name: embedding}}
        self.schema_descriptions = {} # Store descriptions

    def ingest_schema(self, db_id: str, schema_info: Dict[str, List[Dict]]):
        """
        Ingest schema information and compute embeddings for table names and column descriptions.
        schema_info: {table_name: [{name, type, ...}]}
        In a real scenario, we would also use column_meaning.json if available.
        """
        print(f"Ingesting schema for {db_id}...")
        embeddings = {}
        processed_descriptions = {}

        for table, cols in schema_info.items():
            # Create a textual representation of the table for embedding
            # "Table: users. Columns: id, name, email."
            col_names = ", ".join([c['name'] for c in cols])
            text = f"Table: {table}. Columns: {col_names}."
            
            emb = self.model.encode(text)
            embeddings[table] = emb
            processed_descriptions[table] = text
        
        self.schema_embeddings[db_id] = embeddings
        self.schema_descriptions[db_id] = processed_descriptions

    def retrieve_relevant_tables(self, db_id: str, question: str, top_k: int = 5) -> List[str]:
        """
        Retrieve the top-k most relevant tables for a given question.
        """
        if db_id not in self.schema_embeddings:
            return [] # Or raise error

        q_emb = self.model.encode(question)
        
        table_scores = []
        for table, t_emb in self.schema_embeddings[db_id].items():
            # Cosine similarity
            score = np.dot(q_emb, t_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(t_emb))
            table_scores.append((table, score))
        
        # Sort by score descending
        table_scores.sort(key=lambda x: x[1], reverse=True)
        
        return [t[0] for t in table_scores[:top_k]]

if __name__ == "__main__":
    # Test
    rag = RAGService()
    dummy_schema = {
        "users": [{"name": "id"}, {"name": "name"}],
        "orders": [{"name": "id"}, {"name": "user_id"}, {"name": "amount"}],
        "products": [{"name": "id"}, {"name": "label"}]
    }
    rag.ingest_schema("test_db", dummy_schema)
    tables = rag.retrieve_relevant_tables("test_db", "Who bought the most expensive item?")
    print(f"Relevant tables: {tables}")
