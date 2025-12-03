import argparse
import torch

from transformers import (
    T5TokenizerFast,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from src.sqlnet.model import SQLNet
from src.common.data_utils import load_json, read_tables
from src.common.schema_to_text import build_input_text
from src.ratsql_lite.model import RATSQLLite

print(">>> compare_models.py LOADED SUCCESSFULLY")


# -----------------------------------------------------
# DEVICE
# -----------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# -----------------------------------------------------
# Load SQLNet
# -----------------------------------------------------
def load_sqlnet_model(ckpt_path, tokenizer):
    device = get_device()

    model = SQLNet(
        vocab_size=tokenizer.vocab_size,
        pad_idx=tokenizer.pad_token_id,
        emb_dim=256,
        hid_dim=256
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)

    # SQLNet checkpoints were saved as dict
    if "model_state" in state:
        print(">> Extracting SQLNet model_state")
        state = state["model_state"]

    model.load_state_dict(state)
    model.eval()
    return model, device


def sqlnet_generate(model, tokenizer, device, input_text, max_len=80):
    enc = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(device)

    start_id = tokenizer.pad_token_id
    end_id = tokenizer.eos_token_id

    with torch.no_grad():
        out = model.generate(
            enc["input_ids"],
            max_len=max_len,
            start_token_id=start_id,
            end_token_id=end_id
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


# -----------------------------------------------------
# T5 (your fine tuned one)
# -----------------------------------------------------
def t5_generate(model_dir, input_text):
    device = get_device()

    tok = T5TokenizerFast.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir).to(device)
    model.eval()

    enc = tok(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        ids = model.generate(
            enc["input_ids"],
            attention_mask=enc["attention_mask"],
            num_beams=4,
            max_length=150
        )

    return tok.decode(ids[0], skip_special_tokens=True)


# -----------------------------------------------------
# RATSQL-Lite
# -----------------------------------------------------
def load_ratsql_lite(ckpt_path, tokenizer):
    device = get_device()

    model = RATSQLLite(
        vocab_size=tokenizer.vocab_size,
        pad_idx=tokenizer.pad_token_id,
        d_model=256,
        num_heads=4,
        num_layers=4,
        num_relations=4
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)

    if "model_state" in state:
        print(">> Extracting RATSQL-Lite model_state")
        state = state["model_state"]

    model.load_state_dict(state)
    model.eval()
    return model, device


def ratsql_lite_generate(model, tokenizer, device, input_text, max_len=80):
    enc = tokenizer(
        input_text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)

    start_id = tokenizer.pad_token_id
    end_id = tokenizer.eos_token_id

    with torch.no_grad():
        out = model.generate(
            enc["input_ids"],
            max_len=max_len,
            start_token_id=start_id,
            end_token_id=end_id,
            relation_ids=None
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


# -----------------------------------------------------
# GaussAlgo T5-Large (Model 4)
# -----------------------------------------------------
def gaussalgo_generate(question, schema):
    device = get_device()
    model_name = "gaussalgo/T5-LM-Large-text2sql-spider"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    model.eval()

    input_text = build_input_text(question, schema)

    enc = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        out = model.generate(
            enc["input_ids"],
            attention_mask=enc["attention_mask"],
            num_beams=4,
            max_length=150
        )

    return tokenizer.decode(out[0], skip_special_tokens=True)


# -----------------------------------------------------
# MAIN
# -----------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sqlnet_ckpt", required=True)
    parser.add_argument("--t5_model_dir", required=True)
    parser.add_argument("--ratsql_ckpt", required=True)
    parser.add_argument("--tables_json", required=True)
    parser.add_argument("--dev_json", required=True)
    parser.add_argument("--idx", type=int, default=0)
    args = parser.parse_args()

    print("\n=== Running compare_models.py ===")

    # Load Spider example
    dev = load_json(args.dev_json)
    tables = read_tables(args.tables_json)

    ex = dev[args.idx]
    question = ex["question"]
    gold_sql = ex["query"]
    db_id = ex["db_id"]
    schema = tables[db_id]

    input_text = build_input_text(question, schema)

    base_tok = T5TokenizerFast.from_pretrained("t5-base")

    # ---- SQLNet
    sqlnet_model, d1 = load_sqlnet_model(args.sqlnet_ckpt, base_tok)
    pred_sql_sqlnet = sqlnet_generate(sqlnet_model, base_tok, d1, input_text)

    # ---- T5 Fine-Tuned
    pred_sql_t5 = t5_generate(args.t5_model_dir, input_text)

    # ---- RATSQL-Lite
    ratsql_model, d2 = load_ratsql_lite(args.ratsql_ckpt, base_tok)
    pred_sql_ratsql = ratsql_lite_generate(ratsql_model, base_tok, d2, input_text)

    # ---- GaussAlgo T5-Large
    pred_sql_gauss = gaussalgo_generate(question, schema)

    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------
    print("\n============================")
    print(f"DB: {db_id}")
    print("============================")
    print("Question:", question)

    print("\n--- GOLD SQL ---")
    print(gold_sql)

    print("\n--- SQLNet Prediction (LSTM) ---")
    print(pred_sql_sqlnet)

    print("\n--- T5 Fine-Tuned Prediction (Transformer) ---")
    print(pred_sql_t5)

    print("\n--- RATSQL-Lite Prediction (Relation Encoder) ---")
    print(pred_sql_ratsql)

    print("\n--- GaussAlgo T5-LARGE Prediction ---")
    print(pred_sql_gauss)

    print("\n============================\n")


if __name__ == "__main__":
    main()
