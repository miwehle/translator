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

    @classmethod
    def from_preflight_check_result(
        cls,
        check_result: dict,
        *,
        schema_version: int,
        tokenizer_model_name: str,
        src_lang: str,
        tgt_lang: str,
        base_vocab_size: int,
    ) -> "DatasetMetadata":
        return cls(
            schema_version=schema_version,
            tokenizer_model_name=tokenizer_model_name,
            src_lang=src_lang,
            tgt_lang=tgt_lang,
            id_field=check_result["id_field"],
            src_field=check_result["src_field"],
            tgt_field=check_result["tgt_field"],
            base_vocab_size=base_vocab_size,
            src_vocab_size=check_result["src_vocab_size"],
            tgt_vocab_size=check_result["tgt_vocab_size"],
            src_pad_id=check_result["src_pad_idx"],
            tgt_pad_id=check_result["tgt_pad_idx"],
            tgt_bos_id=check_result["inferred_tgt_bos_id"],
            tgt_eos_id=check_result["inferred_tgt_eos_id"],
            num_examples=check_result["num_examples"],
        )
