import requests

class OllamaLLMClient:
    def __init__(self, model="phi3-finetuned"):
        self.model = model
        self.base_url = "http://localhost:11434"
    
    def generate(self, prompt):
        response = requests.post(
            f"{self.base_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False}
        )
        return response.json().get("response", "").strip()
