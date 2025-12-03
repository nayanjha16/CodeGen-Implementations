from torch.utils.data import Dataset
from transformers import T5TokenizerFast
from typing import List, Dict
from torch.utils.data import Dataset
from transformers import T5TokenizerFast
from typing import List, Dict

from src.common.schema_to_text import build_input_text


class SpiderSQLNetDataset(Dataset):
    def __init__(
        self,
        examples: List[Dict],
        tokenizer: T5TokenizerFast, #used to encode question+schema and SQL.
        max_src_len: int = 256,
        max_tgt_len: int = 160,
    ):
        """
        examples: list of dicts {'question', 'query', 'schema'}
        """
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        src_text = build_input_text(ex["question"], ex["schema"])
        tgt_text = ex["query"]

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

        src_ids = src_enc["input_ids"].squeeze(0)  # (src_len,)
        tgt_ids = tgt_enc["input_ids"].squeeze(0)  # (tgt_len,)
        return src_ids, tgt_ids

