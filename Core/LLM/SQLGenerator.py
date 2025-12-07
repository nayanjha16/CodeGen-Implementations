
class SQLGenerator:
    def __init__(self,llm_client): self.llm=llm_client
    def generate_sql(self,question,subgraph):
        schema="\n".join(d['text'] for _,d in subgraph.nodes(data=True))
        prompt=f"Generate only sql query.no explanation.Schema:\n{schema}\nQuestion:{question}\nSQL:"
        return self.llm.generate(prompt)
