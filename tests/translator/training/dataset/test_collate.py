from translator.training.dataset import collate_fn_prod


def test_collate_fn_prod_pads_and_returns_batch_ids():
    batch = [(100, [1, 2, 3], [9, 8]), (101, [4], [7, 6, 5])]

    src, tgt, batch_ids = collate_fn_prod(batch, pad_idx_src=0, pad_idx_tgt=0)

    assert tuple(src.shape) == (2, 3)
    assert tuple(tgt.shape) == (2, 3)
    assert batch_ids == [100, 101]
    assert src.tolist() == [[1, 2, 3], [4, 0, 0]]
    assert tgt.tolist() == [[9, 8, 0], [7, 6, 5]]
