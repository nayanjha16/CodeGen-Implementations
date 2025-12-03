import json
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration


# -----------------------------------------------------------
# Load model + tokenizer
# -----------------------------------------------------------
def load_model(path="src/sql2nosql/checkpoints_t5_sql2mongo"):
    tokenizer = T5Tokenizer.from_pretrained(path)
    model = T5ForConditionalGeneration.from_pretrained(path)

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
# SQL → Mongo translation
# -----------------------------------------------------------
def translate_sql_to_mongo(sql, tokenizer, model, device):
    inputs = tokenizer(sql, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            early_stopping=True,
        )

    text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    cleaned = text.replace("\n", "").replace("\t", "")

    # try JSON conversion
    try:
        return json.loads(cleaned)
    except:
        return cleaned   # fallback raw text


# -----------------------------------------------------------
# Run inference on Spider dev set
# -----------------------------------------------------------
def run_inference(
    spider_path="data/spider/dev_spider.json",
    output_path="src/sql2nosql/data/sql2mongo_dev_predictions.jsonl"
):
    print(f"🔥 Loading Spider dev set: {spider_path}")
    data = json.load(open(spider_path))

    tokenizer, model, device = load_model()

    print(f"🔥 Running inference on {len(data)} examples...")

    with open(output_path, "w") as f:
        for ex in tqdm(data):
            sql = ex["query"]
            db_id = ex["db_id"]

            mongo = translate_sql_to_mongo(sql, tokenizer, model, device)

            f.write(json.dumps({
                "db_id": db_id,
                "sql": sql,
                "mongo": mongo
            }) + "\n")

    print(f"\n✅ Inference complete! Saved to: {output_path}")


# -----------------------------------------------------------
if __name__ == "__main__":
    run_inference()
