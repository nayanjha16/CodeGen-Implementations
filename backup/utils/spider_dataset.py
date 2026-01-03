import os
from datasets import load_dataset

def load_fixed_spider(split: str = "train"):
    """
    Load Turbular/fixed_spider dataset from HuggingFace.

    Each example has:
      - question : natural language question
      - db       : database id
      - query    : ground-truth SQL
      - db_schema: CREATE TABLE statements for that DB
    """
    ds = load_dataset("Turbular/fixed_spider", split=split)
    return ds