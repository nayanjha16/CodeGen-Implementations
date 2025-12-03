import argparse
import torch
from tqdm import tqdm
from transformers import T5TokenizerFast

from src.common.data_utils import load_json, read_tables
from src.common.schema_to_text import build_input_text
from src.sqlnet.model import SQLNet



def load_model(ckpt_path: str, device: torch.device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    tokenizer = T5TokenizerFast.from_pretrained(checkpoint.get("tokenizer_name", "t5-base"))
    pad_idx = checkpoint.get("pad_token_id", tokenizer.pad_token_id)

    model = SQLNet(vocab_size=tokenizer.vocab_size, pad_idx=pad_idx)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--dev_json", default="data/spider/dev_spider.json")
    parser.add_argument("--tables_json", default="data/spider/tables.json")
    parser.add_argument("--num_examples", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer = load_model(args.ckpt, device)
    dev_data = load_json(args.dev_json)
    tables = read_tables(args.tables_json)

    n = min(args.num_examples, len(dev_data))
    correct = 0

    for i in tqdm(range(n), desc="Evaluating"):
        ex = dev_data[i]
        db_id = ex["db_id"]
        question = ex["question"]
        gold_sql = ex["query"].strip()

        schema = tables[db_id]
        inp_text = build_input_text(question, schema)
        enc = tokenizer(
            inp_text,
            max_length=256,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        src_ids = enc["input_ids"].to(device)

        start_id = tokenizer.pad_token_id
        end_id = tokenizer.eos_token_id or tokenizer.pad_token_id

        with torch.no_grad():
            gen_ids = model.generate(src_ids, max_len=160, start_token_id=start_id, end_token_id=end_id)
        pred_sql = tokenizer.decode(gen_ids[0], skip_special_tokens=True).strip()

        if pred_sql == gold_sql:
            correct += 1

    acc = correct / n if n > 0 else 0.0
    print(f"Exact string match on first {n} dev examples: {acc * 100:.2f}% ({correct}/{n})")


if __name__ == "__main__":
    main()
