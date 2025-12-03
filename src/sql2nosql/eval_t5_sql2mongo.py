import json
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

#from sql2nosql.sql_to_mongo_rule_based import SQLtoMongoConverter
#from src.sql2nosql.sql_to_mongo_rule_based import SQLtoMongoConverter
#class SQLToMongoConverter:
from .sql_to_mongo_rule_based import SQLToMongoConverter 




# -----------------------------------------------------------
# Load Model & Tokenizer
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
# SQL → Mongo (T5)
# -----------------------------------------------------------
def translate_sql_to_mongo(sql, tokenizer, model, device):
    inputs = tokenizer(sql, return_tensors="pt", truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            early_stopping=True,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    cleaned = text.replace("\n", "").replace("\t", "").strip()

    try:
        return json.loads(cleaned), True
    except:
        return cleaned, False


# -----------------------------------------------------------
# Loose Structural Match (ignores ordering)
# -----------------------------------------------------------
def structural_match(gold, pred):
    """
    Compare dict/list structures ignoring ordering differences.
    Example:
    gold = [{"$project": {...}}, {"$match": {...}}]
    pred = [{"$match": {...}}, {"$project": {...}}]
    Should return True
    """
    try:
        if isinstance(gold, list) and isinstance(pred, list):
            return sorted(map(str, gold)) == sorted(map(str, pred))
        return gold == pred
    except:
        return False


# -----------------------------------------------------------
# Evaluation
# -----------------------------------------------------------
def evaluate(
    dataset_path="src/sql2nosql/data/sql2mongo_train.jsonl",
    model_path="src/sql2nosql/checkpoints_t5_sql2mongo",
    report_path="src/sql2nosql/eval_results.jsonl"
):
    tokenizer, model, device = load_model(model_path)
    converter = SQLToMongoConverter()

    total = 0
    exact_matches = 0
    structural_matches = 0
    valid_json = 0

    print("🔥 Loading dataset:", dataset_path)
    lines = [json.loads(l) for l in open(dataset_path)]

    print(f"🔥 Evaluating {len(lines)} examples...")

    with open(report_path, "w") as report:
        for ex in tqdm(lines):
            sql = ex["sql"]
            gold_mongo = ex["mongo"]

            # ---- Run T5 model
            pred_mongo, is_json = translate_sql_to_mongo(sql, tokenizer, model, device)

            # ---- Metrics
            total += 1
            if is_json:
                valid_json += 1

            exact = (pred_mongo == gold_mongo)
            if exact:
                exact_matches += 1

            struct = structural_match(gold_mongo, pred_mongo) if is_json else False
            if struct:
                structural_matches += 1

            # ---- Save detailed results
            report.write(json.dumps({
                "sql": sql,
                "gold": gold_mongo,
                "pred": pred_mongo,
                "valid_json": is_json,
                "exact_match": exact,
                "structural_match": struct
            }) + "\n")

    # ---- Summary
    print("\n===== EVALUATION SUMMARY =====")
    print(f"Total examples:            {total}")
    print(f"Valid JSON predictions:    {valid_json} ({valid_json/total*100:.2f}%)")
    print(f"Exact JSON matches:        {exact_matches} ({exact_matches/total*100:.2f}%)")
    print(f"Structural matches:        {structural_matches} ({structural_matches/total*100:.2f}%)")
    print(f"Saved detailed report to:  {report_path}")


# -----------------------------------------------------------
if __name__ == "__main__":
    evaluate()
