
class SchemaRetriever:
    def __init__(self,G,k=10): self.G=G; self.k=k
    def retrieve(self,q,top_k=None):
        top_k=top_k or self.k
        toks=q.lower().split()
        scores=[]
        for n,d in self.G.nodes(data=True):
            s=sum(t in d['text'].lower() for t in toks)
            scores.append((n,s))
        scores=sorted(scores,key=lambda x:x[1],reverse=True)
        nodes=[n for n,_ in scores[:top_k]]
        return nodes, self.G.subgraph(nodes).copy()
