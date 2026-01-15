
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional

class NoSQLConverter:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"):
        """
        Initialize the converter with a model.
        Using the same 0.5B model for efficiency, but valid for larger ones too.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading NoSQL conversion model: {model_name} on {self.device}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        if self.device == "cpu":
            self.model.to(self.device)
            
    def convert_to_mongodb(self, sql_query: str, schema_context: str = "") -> str:
        """
        Convert a SQL query to a MongoDB query (shell syntax).
        """
        prompt = self._build_prompt(sql_query, schema_context)
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=512, 
                temperature=0.1,
                do_sample=False
            )
            
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract the response part
        response = self._extract_response(generated_text)
        return response
        
    def _build_prompt(self, sql: str, schema: str) -> str:
        return f"""### Instruction:
You are an expert database engineer. Convert the following SQL query into a valid MongoDB query (using db.collection.find(), aggregate(), etc.).
Provide ONLY the MongoDB query code. Do not include explanations.

### Context (SQL Schema):
{schema}

### SQL Query:
{sql}

### MongoDB Query:
"""

    def _extract_response(self, text: str) -> str:
        if "### MongoDB Query:" in text:
            parts = text.split("### MongoDB Query:")
            return parts[-1].strip()
        return text.strip()

# Singleton instance
_converter = None

def get_converter():
    global _converter
    if _converter is None:
        _converter = NoSQLConverter()
    return _converter
