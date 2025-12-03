import torch
from transformers import (
    T5TokenizerFast,
    T5ForConditionalGeneration,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
)

from src.sqlnet.model import SQLNet
from src.common.schema_to_text import build_input_text
from src.ratsql_t5_lite.schema_graph_text import build_ratsql_t5_input


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BaseText2SQLModel:
    """Base interface for Text2SQL model wrappers."""
    def predict(self, question: str, schema: dict, db_id: str) -> str:
        raise NotImplementedError


# -------------------------------------------------------------------------
# MODEL 1 — SQLNet (LSTM Baseline)
# -------------------------------------------------------------------------
class SQLNetWrapper(BaseText2SQLModel):
    def __init__(self, ckpt_path: str):
        self.device = get_device()
        self.tokenizer = T5TokenizerFast.from_pretrained("t5-base")

        vocab_size = self.tokenizer.vocab_size
        pad_id = self.tokenizer.pad_token_id

        self.model = SQLNet(
            vocab_size=vocab_size,
            pad_idx=pad_id,
            emb_dim=256,
            hid_dim=256,
        ).to(self.device)

        state = torch.load(ckpt_path, map_location=self.device)
        if "model_state" in state:
            state = state["model_state"]

        self.model.load_state_dict(state)
        self.model.eval()

    def predict(self, question: str, schema: dict, db_id: str) -> str:
        input_text = build_input_text(question, schema)

        enc = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        ).to(self.device)

        start_id = self.tokenizer.pad_token_id
        end_id = self.tokenizer.eos_token_id

        with torch.no_grad():
            gen_ids = self.model.generate(
                enc["input_ids"],
                max_len=80,
                start_token_id=start_id,
                end_token_id=end_id,
            )

        return self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)


# -------------------------------------------------------------------------
# MODEL 2 — T5 Fine-Tuned on Spider
# -------------------------------------------------------------------------
class T5FineTunedWrapper(BaseText2SQLModel):
    def __init__(self, model_dir: str):
        self.device = get_device()
        self.tokenizer = T5TokenizerFast.from_pretrained(model_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    def predict(self, question: str, schema: dict, db_id: str) -> str:
        input_text = build_input_text(question, schema)

        enc = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                enc["input_ids"],
                attention_mask=enc["attention_mask"],
                num_beams=4,
                max_length=150,
                early_stopping=True,
            )

        return self.tokenizer.decode(out[0], skip_special_tokens=True)


# -------------------------------------------------------------------------
# MODEL 3 — NEW RAT-SQL-T5-LITE (Graph + Transformer)
# -------------------------------------------------------------------------
class RATSQLT5LiteWrapper(BaseText2SQLModel):
    def __init__(self, model_dir: str):
        self.device = get_device()
        self.tokenizer = T5TokenizerFast.from_pretrained(model_dir)
        self.model = T5ForConditionalGeneration.from_pretrained(model_dir).to(self.device)
        self.model.eval()

    def predict(self, question: str, schema: dict, db_id: str) -> str:
        input_text = build_ratsql_t5_input(question, schema)

        enc = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=768,
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                enc["input_ids"],
                attention_mask=enc["attention_mask"],
                num_beams=4,
                max_length=160,
                early_stopping=True,
            )

        return self.tokenizer.decode(out[0], skip_special_tokens=True)


# -------------------------------------------------------------------------
# MODEL 4 — GaussAlgo T5-Large
# -------------------------------------------------------------------------
class GaussAlgoT5Wrapper(BaseText2SQLModel):
    def __init__(self, model_name: str = "gaussalgo/T5-LM-Large-text2sql-spider"):
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def predict(self, question: str, schema: dict, db_id: str) -> str:
        input_text = build_input_text(question, schema)

        enc = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                enc["input_ids"],
                attention_mask=enc.get("attention_mask", None),
                num_beams=4,
                max_length=150,
                early_stopping=True,
            )

        return self.tokenizer.decode(out[0], skip_special_tokens=True)
