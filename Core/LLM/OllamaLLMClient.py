import os
import requests


class OllamaLLMClient:
    def __init__(self, model=None, base_url=None):
        # Allow overriding via environment; fall back to defaults.
        self.model = model or os.getenv("OLLAMA_MODEL", "phi3-finetuned")
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    def generate(self, prompt, model=None):
        model_name = model or self.model
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": model_name, "prompt": prompt, "stream": False}
        )
        return response.json().get("response", "").strip()
