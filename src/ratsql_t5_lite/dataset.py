# src/ratsql_t5_lite/dataset.py

from typing import Dict, List
from torch.utils.data import Dataset
from transformers import T5TokenizerFast

from src.common.data_utils import load_json, read_tables
from src.ratsql_t5_lite.schema_graph_text import build_ratsql_t5_input


class SpiderRATGraphT5Dataset(Dataset):
    """
    Dataset for RAT-SQL-Lite-T5:
      - Builds graph-style schema text
      - Uses T5 tokenizer
      - Input: question + schema_graph
      - Target: raw SQL query
    """

    def __init__(
        self,
        train_json_path: str,
        tables_json_path: str,
        tokenizer: T5TokenizerFast,
        max_src_len: int = 512,
        max_tgt_len: int = 160,
    ):
        self.data: List[Dict] = load_json(train_json_path)
        self.tables: Dict = read_tables(tables_json_path)
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        # Filter only examples whose db_id is present in tables.json
        self.examples: List[Dict] = []
        for ex in self.data:
            db_id = ex["db_id"]
            if db_id not in self.tables:
                continue
            self.examples.append(ex)

        print(f"[SpiderRATGraphT5Dataset] Loaded {len(self.examples)} examples")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        ex = self.examples[idx]
        db_id = ex["db_id"]
        question = ex["question"]
        gold_sql = ex["query"]  # raw query string

        schema = self.tables[db_id]
        src_text = build_ratsql_t5_input(question, schema)
        tgt_text = gold_sql

        src_enc = self.tokenizer(
            src_text,
            max_length=self.max_src_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        tgt_enc = self.tokenizer(
            tgt_text,
            max_length=self.max_tgt_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = src_enc["input_ids"].squeeze(0)
        attention_mask = src_enc["attention_mask"].squeeze(0)
        labels = tgt_enc["input_ids"].squeeze(0)

        # Standard T5 convention: mask padding in labels with -100
        pad_id = self.tokenizer.pad_token_id
        labels = labels.clone()
        labels[labels == pad_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
