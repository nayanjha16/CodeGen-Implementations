"""
text2sql_gemini.py

Use Gemini to generate SQL queries on the Turbular/fixed_spider dataset.

Prereqs:
- Set environment variable GEMINI_API_KEY with your Gemini API key.
  On Windows PowerShell:
    $env:GEMINI_API_KEY = "your_api_key_here"
"""

import os
from typing import List, Dict

from datasets import load_dataset
from google import genai
from dotenv import load_dotenv
load_dotenv()


# -------- Gemini helper --------

SYSTEM_INSTRUCTION = """
You are a Text-to-SQL assistant.

Given:
- A database schema (as SQL CREATE TABLE statements)
- A natural language question

Task:
- Generate a single valid SQL query that answers the question.
- Use ONLY tables and columns that exist in the schema.
- Do NOT explain anything, do NOT add comments.
- Output ONLY the SQL query.
"""


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


def init_gemini_client() -> genai.Client:
    """
    Initialize Gemini client.
    The client reads GEMINI_API_KEY from the environment as per official docs.
    """
    # Make sure GEMINI_API_KEY is set
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Get an API key from Google AI Studio and set GEMINI_API_KEY."
        )

    # With google-genai, you normally just need:
    client = genai.Client()
    return client


def generate_sql_with_gemini(
    client: genai.Client,
    question: str,
    db_schema: str,
    gold_sql: str | None = None,
    model_name: str = "gemini-2.5-flash",#gemini-2.0-flash-lite
) -> str:
    """
    Call Gemini to generate a SQL query for the given question + schema.
    """
    prompt = build_prompt(question, db_schema, gold_sql=gold_sql)

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    # response.text is provided by the SDK for convenience
    sql = (response.text or "").strip()
    return sql


# -------- Dataset helper --------

def load_fixed_spider(split: str = "train"):
    """
    Load Turbular/fixed_spider dataset from HuggingFace.

    Each example has:
      - question : natural language question
      - db       : database id
      - query    : ground-truth SQL
      - db_schema: CREATE TABLE statements for that DB
    """
    ds = load_dataset("Turbular/fixed_spider", split=split)
    return ds


def preview_samples(dataset, n: int = 3) -> List[Dict]:
    """
    Return the first n samples as Python dicts (for quick debugging).
    """
    return [dataset[i] for i in range(min(n, len(dataset)))]


# -------- Main script --------

def main():
    # 1. Load dataset
    print("Loading Turbular/fixed_spider dataset from HuggingFace...")
    dataset = load_fixed_spider(split="train")
    print(f"Loaded {len(dataset)} examples.")

    # 2. Init Gemini client
    print("Initializing Gemini client...")
    client = init_gemini_client()
    print("Gemini client ready.")

    # 3. Use a small subset for demo (e.g. first 5 examples)
    num_examples = 5
    print(f"\nGenerating SQL for first {num_examples} examples...\n")    
    print

    for idx in range(num_examples):
        example = dataset[idx]
        question = example["question"]
        db_id = example["db"]
        gold_sql = example["query"]
        db_schema = example["db_schema"]

        print("=" * 80)
        print(f"Example #{idx}")
        print(f"DB ID    : {db_id}")
        print(f"Question : {question}")
        print("-" * 80)
        print("Ground truth SQL from dataset:")
        print(gold_sql)
        print("-" * 80)

        try:
            predicted_sql = generate_sql_with_gemini(
                client=client,
                question=question,
                db_schema=db_schema,
                gold_sql=gold_sql,
                model_name="gemini-2.5-flash",  # or gemini-2.5-pro if you have access
            )
        except Exception as e:
            print(f"[ERROR] Gemini call failed: {e}")
            continue

        print("Gemini predicted SQL:")
        print(predicted_sql)
        print("=" * 80)
        print()

    print("Done.")


if __name__ == "__main__":
    main()
