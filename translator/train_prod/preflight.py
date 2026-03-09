from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def validate_records_contract(
    records: Iterable[Any],
    *,
    id_field: str = "id",
    src_field: str = "src_ids",
    tgt_field: str = "tgt_ids",
    require_unique_ids: bool = True,
    min_seq_len: int = 2,
) -> dict[str, Any]:
    seen_ids: set[int] = set()
    total = 0
    min_src_len = None
    min_tgt_len = None
    max_src_len = 0
    max_tgt_len = 0
    max_src_token = -1
    max_tgt_token = -1
    bos_candidates: dict[int, int] = {}
    eos_candidates: dict[int, int] = {}

    for item in records:
        total += 1
        if not isinstance(item, Mapping):
            raise ValueError(f"Record #{total} is not a mapping/object.")
        if id_field not in item or src_field not in item or tgt_field not in item:
            raise ValueError(
                f"Missing required fields in record #{total}: "
                f"expected '{id_field}', '{src_field}', '{tgt_field}'."
            )

        ex_id = int(item[id_field])
        src = item[src_field]
        tgt = item[tgt_field]
        if not isinstance(src, list) or not isinstance(tgt, list):
            raise ValueError(
                f"Record id={ex_id} has invalid types: "
                f"{src_field}/{tgt_field} must be list[int]."
            )
        if len(src) < min_seq_len or len(tgt) < min_seq_len:
            raise ValueError(
                f"Record id={ex_id} violates min_seq_len={min_seq_len}: "
                f"len(src)={len(src)} len(tgt)={len(tgt)}."
            )

        try:
            src_ids = [int(x) for x in src]
            tgt_ids = [int(x) for x in tgt]
        except Exception as exc:
            raise ValueError(
                f"Record id={ex_id} contains non-integer token IDs."
            ) from exc

        if require_unique_ids:
            if ex_id in seen_ids:
                raise ValueError(f"Duplicate id detected: id={ex_id}.")
            seen_ids.add(ex_id)

        src_len = len(src_ids)
        tgt_len = len(tgt_ids)
        min_src_len = src_len if min_src_len is None else min(min_src_len, src_len)
        min_tgt_len = tgt_len if min_tgt_len is None else min(min_tgt_len, tgt_len)
        max_src_len = max(max_src_len, src_len)
        max_tgt_len = max(max_tgt_len, tgt_len)
        max_src_token = max(max_src_token, max(src_ids))
        max_tgt_token = max(max_tgt_token, max(tgt_ids))

        bos = tgt_ids[0]
        eos = tgt_ids[-1]
        bos_candidates[bos] = bos_candidates.get(bos, 0) + 1
        eos_candidates[eos] = eos_candidates.get(eos, 0) + 1

    if total == 0:
        raise ValueError("Dataset is empty.")

    inferred_bos = max(bos_candidates, key=lambda tok: bos_candidates[tok])
    inferred_eos = max(eos_candidates, key=lambda tok: eos_candidates[tok])
    bos_consistency = bos_candidates[inferred_bos] / total
    eos_consistency = eos_candidates[inferred_eos] / total

    return {
        "num_examples": total,
        "min_src_len": min_src_len,
        "max_src_len": max_src_len,
        "min_tgt_len": min_tgt_len,
        "max_tgt_len": max_tgt_len,
        "max_src_token_id": max_src_token,
        "max_tgt_token_id": max_tgt_token,
        "inferred_tgt_bos_id": inferred_bos,
        "inferred_tgt_eos_id": inferred_eos,
        "bos_consistency": bos_consistency,
        "eos_consistency": eos_consistency,
    }
