from openai import OpenAI
import os

class OpenAIClient:
    def __init__(self, model="gpt-4o-mini"):
        self.model=model
        self.client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    def generate(self,prompt):
        resp=self.client.chat.completions.create(
            model=self.model,
            messages=[{"role":"user","content":prompt}]
        )
        return resp.choices[0].message["content"]
