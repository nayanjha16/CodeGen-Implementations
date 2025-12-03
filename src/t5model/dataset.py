from typing import List, Dict
from torch.utils.data import Dataset
from transformers import T5TokenizerFast

from src.common.data_utils import load_json, read_tables
from src.common.schema_to_text import build_input_text


class SpiderText2SQLDataset(Dataset):
    """
    Text-to-SQL dataset for T5:
    Input  → "translate English to SQL: ... schema: ..."
    Output → SQL query string
    """

    def __init__(
        self,
        json_path: str,
        tables_json_path: str,
        tokenizer: T5TokenizerFast,
        max_input_len: int = 256,
        max_output_len: int = 160,
    ):
        self.data = load_json(json_path)
        self.tables = read_tables(tables_json_path)
        self.tokenizer = tokenizer

        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

        self.examples = [
            ex for ex in self.data if ex["db_id"] in self.tables
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        db_id = ex["db_id"]

        question = ex["question"]
        sql = ex["query"]
        schema = self.tables[db_id]

        input_text = build_input_text(question, schema)

        enc = self.tokenizer(
            input_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_input_len,
            return_tensors="pt",
        )

        with self.tokenizer.as_target_tokenizer():
            tgt = self.tokenizer(
                sql,
                padding="max_length",
                truncation=True,
                max_length=self.max_output_len,
                return_tensors="pt",
            )

        labels = tgt["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": labels,
        }
