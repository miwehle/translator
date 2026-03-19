from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from ..types import Example


def _format_example_ids(example_ids: list[int], limit: int = 5) -> str:
    shown = example_ids[:limit]
    suffix = "" if len(example_ids) <= limit else ", ..."
    return f"{shown}{suffix}"


def check_dataset(
    examples: Iterable[Example],
    *,
    id_field: str = "id",
    src_field: str = "src_ids",
    tgt_field: str = "tgt_ids",
    src_pad_idx: int | None = None,
    tgt_pad_idx: int | None = None,
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
    bos_example_ids: dict[int, list[int]] = {}
    eos_example_ids: dict[int, list[int]] = {}
    src_pad_example_ids: list[int] = []
    tgt_pad_example_ids: list[int] = []

    for item in examples:
        total += 1
        if not isinstance(item, Mapping):
            raise ValueError(f"Example #{total} is not a mapping/object.")
        if id_field not in item or src_field not in item or tgt_field not in item:
            raise ValueError(
                f"Missing required fields in example #{total}: "
                f"expected '{id_field}', '{src_field}', '{tgt_field}'."
            )

        ex_id = int(item[id_field])
        src = item[src_field]
        tgt = item[tgt_field]
        if not isinstance(src, list) or not isinstance(tgt, list):
            raise ValueError(
                f"Example id={ex_id} has invalid types: "
                f"{src_field}/{tgt_field} must be list[int]."
            )
        if len(src) < min_seq_len or len(tgt) < min_seq_len:
            raise ValueError(
                f"Example id={ex_id} violates min_seq_len={min_seq_len}: "
                f"len(src)={len(src)} len(tgt)={len(tgt)}."
            )

        try:
            src_ids = [int(x) for x in src]
            tgt_ids = [int(x) for x in tgt]
        except Exception as exc:
            raise ValueError(
                f"Example id={ex_id} contains non-integer token IDs."
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
        bos_example_ids.setdefault(bos, []).append(ex_id)
        eos_example_ids.setdefault(eos, []).append(ex_id)
        if src_pad_idx is not None and int(src_pad_idx) in src_ids:
            src_pad_example_ids.append(ex_id)
        if tgt_pad_idx is not None and int(tgt_pad_idx) in tgt_ids:
            tgt_pad_example_ids.append(ex_id)

    if total == 0:
        raise ValueError("Dataset is empty.")

    inferred_bos = max(bos_candidates, key=lambda tok: bos_candidates[tok])
    inferred_eos = max(eos_candidates, key=lambda tok: eos_candidates[tok])
    bos_consistency = bos_candidates[inferred_bos] / total
    eos_consistency = eos_candidates[inferred_eos] / total

    if bos_consistency < 1.0:
        inconsistent_bos_ids: list[int] = []
        for bos_token, example_ids in bos_example_ids.items():
            if bos_token == inferred_bos:
                continue
            inconsistent_bos_ids.extend(example_ids)
        raise ValueError(
            "Target BOS token is not consistent across dataset: "
            f"expected all {tgt_field} sequences to start with {inferred_bos}, "
            f"but found mismatches in example ids "
            f"{_format_example_ids(inconsistent_bos_ids)} "
            f"(consistency={bos_consistency:.2%})."
        )

    if eos_consistency < 1.0:
        inconsistent_eos_ids: list[int] = []
        for eos_token, example_ids in eos_example_ids.items():
            if eos_token == inferred_eos:
                continue
            inconsistent_eos_ids.extend(example_ids)
        raise ValueError(
            "Target EOS token is not consistent across dataset: "
            f"expected all {tgt_field} sequences to end with {inferred_eos}, "
            f"but found mismatches in example ids "
            f"{_format_example_ids(inconsistent_eos_ids)} "
            f"(consistency={eos_consistency:.2%})."
        )

    if inferred_bos == inferred_eos:
        raise ValueError(
            "Target BOS and EOS token must differ: "
            f"inferred_bos_id={inferred_bos} inferred_eos_id={inferred_eos}."
        )

    if src_pad_idx is not None and src_pad_example_ids:
        raise ValueError(
            "Source pad token must not appear in raw source sequences: "
            f"src_pad_idx={src_pad_idx} found in example ids "
            f"{_format_example_ids(src_pad_example_ids)}."
        )

    if tgt_pad_idx is not None and tgt_pad_example_ids:
        raise ValueError(
            "Target pad token must not appear in raw target sequences: "
            f"tgt_pad_idx={tgt_pad_idx} found in example ids "
            f"{_format_example_ids(tgt_pad_example_ids)}."
        )

    resolved_src_pad_idx = (
        src_pad_idx
        if src_pad_idx is not None
        else max_src_token + 1
    )
    resolved_tgt_pad_idx = (
        tgt_pad_idx
        if tgt_pad_idx is not None
        else max_tgt_token + 1
    )
    tgt_sos_idx = inferred_bos
    src_vocab_size = max(max_src_token, int(resolved_src_pad_idx)) + 1
    tgt_vocab_size = max(max_tgt_token, int(resolved_tgt_pad_idx), tgt_sos_idx) + 1
    return {
        "id_field": id_field,
        "src_field": src_field,
        "tgt_field": tgt_field,
        "src_pad_idx": resolved_src_pad_idx,
        "tgt_pad_idx": resolved_tgt_pad_idx,
        "tgt_sos_idx": tgt_sos_idx,
        "src_vocab_size": src_vocab_size,
        "tgt_vocab_size": tgt_vocab_size,
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
