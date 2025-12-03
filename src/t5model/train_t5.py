import os
import argparse

import torch
from torch.utils.data import DataLoader
from transformers import (
    T5ForConditionalGeneration,
    T5TokenizerFast,
    get_linear_schedule_with_warmup,
)

from torch.optim import AdamW
from tqdm import tqdm

from src.t5model.dataset import SpiderText2SQLDataset


def device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def collate(batch):
    keys = batch[0].keys()
    output = {}
    for k in keys:
        output[k] = torch.stack([b[k] for b in batch])
    return output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_json", required=True)
    parser.add_argument("--dev_json", required=False)
    parser.add_argument("--tables_json", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--model_name", default="t5-base")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dev = device()
    print("Device:", dev)

    tokenizer = T5TokenizerFast.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    model.to(dev)

    train_dataset = SpiderText2SQLDataset(
        args.train_json, args.tables_json, tokenizer
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )

    optim = AdamW(model.parameters(), lr=args.lr)

    steps = len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=int(steps * 0.1),
        num_training_steps=steps * args.epochs,
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in pbar:
            batch = {k: v.to(dev) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss

            loss.backward()
            optim.step()
            scheduler.step()
            optim.zero_grad()

            pbar.set_postfix({"loss": float(loss)})

        ckpt = os.path.join(args.output_dir, f"epoch{epoch}")
        model.save_pretrained(ckpt)
        tokenizer.save_pretrained(ckpt)
        print("Saved:", ckpt)


if __name__ == "__main__":
    main()
