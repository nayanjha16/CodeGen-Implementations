import argparse
import torch
from tqdm import tqdm
from transformers import T5TokenizerFast, T5ForConditionalGeneration

from src.common.data_utils import read_tables, load_json
from src.common.schema_to_text import build_input_text


def normalize(sql):
    return " ".join(sql.lower().strip().split())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--tables_json", required=True)
    p.add_argument("--dev_json", required=True)
    p.add_argument("--max_examples", type=int, default=200)
    args = p.parse_args()

    tokenizer = T5TokenizerFast.from_pretrained(args.model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir)
    model.eval()

    dev_data = load_json(args.dev_json)
    tables = read_tables(args.tables_json)

    correct = 0
    n = min(args.max_examples, len(dev_data))

    for i in tqdm(range(n), desc="Evaluating"):
        ex = dev_data[i]
        db_id = ex["db_id"]

        input_text = build_input_text(ex["question"], tables[db_id])

        inp = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        out_ids = model.generate(inp["input_ids"], num_beams=4, max_length=160)
        pred = tokenizer.decode(out_ids[0], skip_special_tokens=True)

        if normalize(pred) == normalize(ex["query"]):
            correct += 1

    print(f"Exact Match: {correct}/{n} = {100*correct/n:.2f}%")


if __name__ == "__main__":
    main()
