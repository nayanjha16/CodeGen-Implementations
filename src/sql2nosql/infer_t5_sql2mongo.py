import json
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


# -----------------------------------------------------------
# Load model + tokenizer
# -----------------------------------------------------------
def load_model(path="src/sql2nosql/checkpoints_t5_sql2mongo"):
    tokenizer = T5Tokenizer.from_pretrained(path)
    model = T5ForConditionalGeneration.from_pretrained(path)

    # auto-select device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    model.to(device)
    model.eval()

    return tokenizer, model, device


# -----------------------------------------------------------
# Convert SQL → Mongo using trained T5 model
# -----------------------------------------------------------
def translate_sql_to_mongo(sql: str, tokenizer, model, device):
    # Prepare input
    inputs = tokenizer(sql, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Inference
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            early_stopping=True,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    text = text.strip()

    # Try to parse JSON
    try:
        cleaned = text.replace("\n", "").replace("\t", "")
        return json.loads(cleaned)
    except Exception:
        # Return raw text if JSON is malformed
        return text


# -----------------------------------------------------------
# CLI test
# -----------------------------------------------------------
if __name__ == "__main__":
    tokenizer, model, device = load_model()
    sql = "SELECT name FROM student WHERE score > 90 LIMIT 5"
    mongo = translate_sql_to_mongo(sql, tokenizer, model, device)

    print("\nInput SQL:")
    print(sql)

    print("\nGenerated MongoDB Pipeline:")
    print(mongo)
