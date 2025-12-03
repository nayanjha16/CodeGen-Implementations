# src/gaussalgo_t5/model.py

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class GaussAlgoT5Text2SQL:
    """
    Wrapper for gaussalgo/T5-LM-Large-text2sql-spider.
    This is your Model 4: a strong, publicly available Text2SQL model trained on Spider.
    """

    def __init__(self, model_name: str = "gaussalgo/T5-LM-Large-text2sql-spider"):
        self.device = get_device()
        print(f"[GaussAlgo T5] Loading model: {model_name}")
        print(f"[GaussAlgo T5] Using device: {self.device}")

        # Load tokenizer + model from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def generate(self, input_text: str, max_length: int = 200, num_beams: int = 4) -> str:
        """
        Generate SQL from input_text (which should include question + schema).
        """
        enc = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512,
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                enc["input_ids"],
                attention_mask=enc.get("attention_mask", None),
                max_length=max_length,
                num_beams=num_beams,
                early_stopping=True,
            )

        return self.tokenizer.decode(out[0], skip_special_tokens=True)
