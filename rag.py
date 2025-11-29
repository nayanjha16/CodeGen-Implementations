
import faiss, numpy as np
from sentence_transformers import SentenceTransformer

class RAGRetriever:
    def __init__(self, dataset):
        self.dataset = dataset
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        corpus = [d["question"] + " " + d["schema"] for d in dataset]
        embeddings = self.embedder.encode(corpus)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(np.array(embeddings))

    def retrieve(self, query, k=1):
        q_emb = self.embedder.encode([query])
        _, idx = self.index.search(np.array(q_emb), k)
        return self.dataset[idx[0][0]]
