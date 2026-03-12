from translator.train_prod import check_examples


def test_check_examples_collects_basic_stats():
    rows = [
        {"id": 1, "src_ids": [10, 11], "tgt_ids": [2, 20, 3]},
        {"id": 2, "src_ids": [12, 13, 14], "tgt_ids": [2, 21, 22, 3]},
    ]

    out = check_examples(rows, src_pad_idx=98, tgt_pad_idx=99)

    assert out["num_examples"] == 2
    assert out["min_src_len"] == 2
    assert out["max_src_len"] == 3
    assert out["min_tgt_len"] == 3
    assert out["max_tgt_len"] == 4
    assert out["max_src_token_id"] == 14
    assert out["max_tgt_token_id"] == 22
    assert out["inferred_tgt_bos_id"] == 2
    assert out["inferred_tgt_eos_id"] == 3
    assert out["bos_consistency"] == 1.0
    assert out["eos_consistency"] == 1.0


def test_check_examples_rejects_duplicate_ids():
    rows = [
        {"id": 1, "src_ids": [1, 2], "tgt_ids": [2, 3]},
        {"id": 1, "src_ids": [4, 5], "tgt_ids": [2, 3]},
    ]

    try:
        check_examples(rows)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Duplicate id" in str(exc)


def test_check_examples_rejects_inconsistent_tgt_bos():
    rows = [
        {"id": 1, "src_ids": [10, 11], "tgt_ids": [2, 20, 3]},
        {"id": 2, "src_ids": [12, 13], "tgt_ids": [7, 21, 3]},
    ]

    try:
        check_examples(rows, src_pad_idx=98, tgt_pad_idx=99)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "BOS token is not consistent" in str(exc)
        assert "id" in str(exc)


def test_check_examples_rejects_inconsistent_tgt_eos():
    rows = [
        {"id": 1, "src_ids": [10, 11], "tgt_ids": [2, 20, 3]},
        {"id": 2, "src_ids": [12, 13], "tgt_ids": [2, 21, 8]},
    ]

    try:
        check_examples(rows, src_pad_idx=98, tgt_pad_idx=99)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "EOS token is not consistent" in str(exc)
        assert "id" in str(exc)


def test_check_examples_rejects_equal_tgt_bos_and_eos():
    rows = [
        {"id": 1, "src_ids": [10, 11], "tgt_ids": [2, 20, 2]},
        {"id": 2, "src_ids": [12, 13], "tgt_ids": [2, 21, 2]},
    ]

    try:
        check_examples(rows, src_pad_idx=98, tgt_pad_idx=99)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "BOS and EOS token must differ" in str(exc)


def test_check_examples_rejects_pad_token_in_raw_tgt_sequences():
    rows = [
        {"id": 1, "src_ids": [10, 11], "tgt_ids": [2, 99, 3]},
        {"id": 2, "src_ids": [12, 13], "tgt_ids": [2, 21, 3]},
    ]

    try:
        check_examples(rows, src_pad_idx=98, tgt_pad_idx=99)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "pad token must not appear" in str(exc)
        assert "id" in str(exc)


def test_check_examples_rejects_pad_token_in_raw_src_sequences():
    rows = [
        {"id": 1, "src_ids": [10, 98], "tgt_ids": [2, 20, 3]},
        {"id": 2, "src_ids": [12, 13], "tgt_ids": [2, 21, 3]},
    ]

    try:
        check_examples(rows, src_pad_idx=98, tgt_pad_idx=99)
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "Source pad token must not appear" in str(exc)
        assert "id" in str(exc)
