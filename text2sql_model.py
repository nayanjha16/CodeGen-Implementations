
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.sql_cleaner import extract_last_sql

class Text2SQLModel:
    def __init__(self, model_id="Qwen/Qwen2.5-1.5B-Instruct"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16
        )
        print(f"[MODEL] Loaded {model_id}")

    def generate_sql(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        output = self.model.generate(**inputs, max_new_tokens=150, do_sample=False)
        decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return extract_last_sql(decoded)
