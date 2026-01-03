import os
import google.generativeai as genai
from google import genai
from dotenv import load_dotenv
from utils.prompt import build_prompt


load_dotenv()

class GeminiClient:
    """
    Manages the initialization and interaction with the Gemini API.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash"):
        """
        Initializes the Gemini client and checks for the API key.
        """
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY environment variable is not set. "
                "Get an API key from Google AI Studio and set GEMINI_API_KEY."
            )
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
    
    def generate_sql(
        self,
        question: str,
        db_schema: str,
        gold_sql: str | None = None,
    ) -> str:
        """
        Call Gemini to generate a SQL query for the given question + schema.
        """
        prompt = build_prompt(question, db_schema, gold_sql=gold_sql)

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )

        # response.text is provided by the SDK for convenience
        sql = (response.text or "").strip()
        return sql

    