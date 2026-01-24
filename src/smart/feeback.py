from urllib import response
from google import genai
import json
import os
import re

def predict_schema(nlq, db_schema,sqlQuery,nosqlQuery):
    api_key = os.environ.get('GOOGLE_API_KEY', '')  # Fallback for local dev
    client = genai.Client(api_key=api_key)
    
        
    # SMART-style prompt for schema prediction
    prompt = f"""
        Task: SMART Framework NoSQL Generator
        Model: MongoDB 3.6

        Context (Database Schema):
        {json.dumps(db_schema, indent=2)}
        sqlQuery: "{sqlQuery}"
        nosqlQuery: "{nosqlQuery}"
        Input Query:
        "{nlq}"

        Instructions:
        1. compare the sqlQuery and nosqlQuery.
        2. strictly use Context (Database Schema) to
        3. update the nosqlQuery to match the sqlQuery results.use only valid mongodb 3.6 syntax.
        4. result_fields: Identify fields to be projected in the final output.
        5. mongodb_query: Generate a JSON object containing "collection" (string) and "pipeline" (list of stages).

        Constraint:
        - Output MUST be valid JSON.
        - No preamble, no markdown code blocks, no explanation.
        - For MongoDB 3.6 compatibility, avoid using modern operators like $merge or $set or $toDouble.
    """
    response = client.models.generate_content(
        model="gemini-pro-latest", # Changed to gemini-pro-latest as it is available
        contents=prompt
        # Removed temperature=0 as it's not supported by genai.Client.models.generate_content
    )
    return response

   