import os
SYSTEM_INSTRUCTION = """You are an expert SQL generator."""

def build_prompt(question: str, db_schema: str, gold_sql: str | None = None) -> str:
    """
    Build the text prompt that will be sent to Gemini.
    gold_sql is optional (only for reference / training-style prompts).
    """
    prompt = f"""{SYSTEM_INSTRUCTION}

    DATABASE SCHEMA:
    {db_schema}

    QUESTION:
    {question}
    """
    if gold_sql:
        prompt += f"""

    (For reference, here is the original ground-truth SQL used in the dataset:
    {gold_sql}
    Do NOT just copy it. Instead, generate a correct SQL query yourself.)
    """
    return prompt