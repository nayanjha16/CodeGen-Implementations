from google import genai
import json
import os
import re

class SLM:
    def __init__(self):
        pass

    def predict_schema(self, nlq, db_schema):
        api_key = os.environ.get('GOOGLE_API_KEY', '')  # Fallback for local dev
        client = genai.Client(api_key=api_key)
        
         
        # SMART-style prompt for schema prediction
        prompt = f"""
            Task: SMART Framework NoSQL Generator
            Model: MongoDB 3.6

            Context (Database Schema):
            {json.dumps(db_schema, indent=2)}

            Input Query:
            "{nlq}"

            Instructions:
            1. database_fields: Identify fields present in the schema used for filtering.
            2. non_database_fields: Identify entities mentioned not found in the schema.
            3. result_fields: Identify fields to be projected in the final output.
            4. mongodb_query: Generate a JSON object containing "collection" (string) and "pipeline" (list of stages).

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

    

   