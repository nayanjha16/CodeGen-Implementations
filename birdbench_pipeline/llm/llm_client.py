import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_llm(prompt: str, model: str = "gpt-4o-mini") -> str:
    """
    Call OpenAI LLM and return raw text response.
    """
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are an expert SQL to MongoDB translator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )
    return response.choices[0].message.content
