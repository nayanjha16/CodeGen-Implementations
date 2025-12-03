import os
import argparse
import random
from typing import List, Dict

import torch
from torch.utils.data import DataLoader
from torch import nn
from tqdm import tqdm
from transformers import T5TokenizerFast

from src.common.data_utils import load_json, read_tables
from src.sqlnet.dataset import SpiderSQLNetDataset
from src.sqlnet.model import SQLNet


def build_examples(train_json_path: str, tables_json_path: str) -> List[Dict]:
    train_data = load_json(train_json_path)
    tables = read_tables(tables_json_path)
    examples = []
    for ex in train_data:
        db_id = ex["db_id"]
        if db_id not in tables:
            continue
        examples.append(
            {
                "question": ex["question"],
                "query": ex["query"],
                "schema": tables[db_id],
            }
        )
    return examples


def collate_fn(batch):
    src_batch = torch.stack([b[0] for b in batch], dim=0)
    tgt_batch = torch.stack([b[1] for b in batch], dim=0)
    return src_batch, tgt_batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", default="data/spider/train_spider.json")
    parser.add_argument("--tables_json", default="data/spider/tables.json")
    parser.add_argument("--save_dir", default="models/sqlnet")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--max_src_len", type=int, default=256)
    parser.add_argument("--max_tgt_len", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)

    tokenizer = T5TokenizerFast.from_pretrained("t5-base") #tokenizer
    pad_idx = tokenizer.pad_token_id

    examples = build_examples(args.train_json, args.tables_json)
    dataset = SpiderSQLNetDataset(examples, tokenizer, args.max_src_len, args.max_tgt_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SQLNet(vocab_size=tokenizer.vocab_size, pad_idx=pad_idx).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for src_ids, tgt_ids in pbar:
            src_ids = src_ids.to(device)
            tgt_ids = tgt_ids.to(device)

            optimizer.zero_grad()
            logits = model(src_ids, tgt_ids, teacher_forcing=0.5)  # (batch, tgt_len, vocab)
            # Shift to align predictions & targets
            logits_flat = logits[:, 1:, :].reshape(-1, logits.size(-1))
            targets_flat = tgt_ids[:, 1:].reshape(-1)
            loss = criterion(logits_flat, targets_flat)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=epoch_loss / (pbar.n + 1))

        ckpt_path = os.path.join(args.save_dir, f"sqlnet_epoch{epoch}.pt")
        torch.save(model.state_dict(), save_path)

        print(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
