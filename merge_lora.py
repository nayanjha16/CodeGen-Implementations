from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

print("Loading base model (microsoft/Phi-3-mini-4k-instruct)...")
base_model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3-mini-4k-instruct",
    torch_dtype=torch.float16,
    trust_remote_code=True
)

print("Loading LoRA adapter from ./final_model...")
model = PeftModel.from_pretrained(base_model, "./final_model")

print("Merging adapter with base model...")
merged_model = model.merge_and_unload()

print("Saving merged model to ./merged_model...")
merged_model.save_pretrained("./merged_model")

print("Saving tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("./final_model")
tokenizer.save_pretrained("./merged_model")

print("Done! Merged model saved to ./merged_model")
