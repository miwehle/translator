from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class DatasetMetadata:
    schema_version: int
    tokenizer_model_name: str
    src_lang: str
    tgt_lang: str
    id_field: str
    src_field: str
    tgt_field: str
    base_vocab_size: int
    src_vocab_size: int
    tgt_vocab_size: int
    src_pad_id: int
    tgt_pad_id: int
    tgt_bos_id: int
    tgt_eos_id: int
    num_examples: int

    @classmethod
    def from_file(cls, path: str | Path) -> "DatasetMetadata":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(**yaml.safe_load(handle))
