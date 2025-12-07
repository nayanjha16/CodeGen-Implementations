from typing import Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

def load_model(model_name: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct") -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Loads the Qwen2.5-Coder-0.5B-Instruct model and tokenizer.
    
    Args:
        model_name: HuggingFace model identifier (default: Qwen/Qwen2.5-Coder-0.5B-Instruct)
        
    Returns:
        A tuple of (model, tokenizer)
        
    Raises:
        Exception: If model loading fails
    """
    print(f"Loading model: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e
