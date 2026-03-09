from translator.train_prod import validate_records_contract


def test_validate_records_contract_collects_basic_stats():
    rows = [
        {"id": 1, "src_ids": [10, 11], "tgt_ids": [2, 20, 3]},
        {"id": 2, "src_ids": [12, 13, 14], "tgt_ids": [2, 21, 22, 3]},
    ]

    out = validate_records_contract(rows)

    assert out["num_examples"] == 2
    assert out["min_src_len"] == 2
    assert out["max_src_len"] == 3
    assert out["min_tgt_len"] == 3
    assert out["max_tgt_len"] == 4
    assert out["max_src_token_id"] == 14
    assert out["max_tgt_token_id"] == 22
    assert out["inferred_tgt_bos_id"] == 2
    assert out["inferred_tgt_eos_id"] == 3


def test_validate_records_contract_rejects_duplicate_ids():
    rows = [
        {"id": 1, "src_ids": [1, 2], "tgt_ids": [2, 3]},
        {"id": 1, "src_ids": [4, 5], "tgt_ids": [2, 3]},
    ]

    try:
        validate_records_contract(rows)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Duplicate id" in str(exc)
