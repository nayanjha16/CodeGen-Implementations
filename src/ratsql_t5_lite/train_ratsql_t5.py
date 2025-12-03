# src/ratsql_t5_lite/train_ratsql_t5.py

import os
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import T5ForConditionalGeneration, T5TokenizerFast, get_linear_schedule_with_warmup

from src.ratsql_t5_lite.dataset import SpiderRATGraphT5Dataset


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collate_fn(batch):
    import torch
    input_ids = torch.stack([b["input_ids"] for b in batch], dim=0)
    attention_mask = torch.stack([b["attention_mask"] for b in batch], dim=0)
    labels = torch.stack([b["labels"] for b in batch], dim=0)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--tables_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--model_name", type=str, default="t5-base")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")

    tokenizer = T5TokenizerFast.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name).to(device)

    dataset = SpiderRATGraphT5Dataset(
        train_json_path=args.train_json,
        tables_json_path=args.tables_json,
        tokenizer=tokenizer,
        max_src_len=512,
        max_tgt_len=160,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.lr)

    num_training_steps = len(dataloader) * args.epochs
    num_warmup_steps = int(0.1 * num_training_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps
    )

    model.train()
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_loss += loss.item()
            pbar.set_postfix({"loss": epoch_loss / max(1, pbar.n)})

        # Save at end of each epoch
        save_dir = os.path.join(args.output_dir, f"epoch{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        model.save_pretrained(save_dir)
        tokenizer.save_pretrained(save_dir)
        print(f"Saved RAT-SQL-Lite-T5 checkpoint to: {save_dir}")


if __name__ == "__main__":
    main()
