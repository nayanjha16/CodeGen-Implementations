import argparse
import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration

from src.common.data_utils import load_json, read_tables
from src.common.schema_to_text import build_input_text


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)
    p.add_argument("--tables_json", required=True)
    p.add_argument("--dev_json", required=True)
    p.add_argument("--idx", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    dev = device()

    tokenizer = T5TokenizerFast.from_pretrained(args.model_dir)
    model = T5ForConditionalGeneration.from_pretrained(args.model_dir).to(dev)

    dev_data = load_json(args.dev_json)
    tables = read_tables(args.tables_json)

    ex = dev_data[args.idx]

    db_id = ex["db_id"]
    input_text = build_input_text(ex["question"], tables[db_id])

    inp = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(dev)

    out_ids = model.generate(
        inp["input_ids"], 
        attention_mask=inp["attention_mask"], 
        num_beams=4,
        max_length=160
    )

    pred_sql = tokenizer.decode(out_ids[0], skip_special_tokens=True)

    print("DB:", db_id)
    print("Q:", ex["question"])
    print("Gold:", ex["query"])
    print("Pred:", pred_sql)


if __name__ == "__main__":
    main()
