import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from typing import Tuple, Optional
import sys
import os

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config

def load_model(adapter_path: Optional[str] = None, model_name: str = config.MODEL_NAME) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Loads the Qwen2.5-Coder model and tokenizer.
    
    Uses configuration from config.py.
    Supports 4-bit quantization if enabled in config.
    If adapter_path is provided, loads the LoRA adapter.
    
    Args:
        adapter_path: Path to the LoRA adapter checkpoint (optional)
        model_name: Name of the model to load. Defaults to config.MODEL_NAME.
    
    Returns:
        model: The loaded causal language model (wrapped with PeftModel if adapter provided)
        tokenizer: The loaded tokenizer
    """
    print(f"Loading model: {model_name}...")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Configure quantization if enabled
        if config.USE_4BIT_QUANTIZATION and config.DEVICE == "cuda":
            print("Using 4-bit quantization (QLoRA configuration)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            print(f"Loading model on {config.DEVICE} (no quantization)...")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if config.DEVICE == "cuda" else None,
                trust_remote_code=True
            )
            if config.DEVICE == "cpu":
                model = model.to("cpu")

        if adapter_path:
            print(f"Loading adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)
            print("Adapter loaded successfully.")

        print("Model loaded successfully.")
        return model, tokenizer
        
    except Exception as e:
        print(f"Error loading model: {e}")
        raise e
