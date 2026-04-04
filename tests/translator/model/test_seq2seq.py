from __future__ import annotations

import torch

from translator.model import Seq2Seq


class TestSeq2Seq:
    def test_translate_returns_decode_ready_ids_without_initial_tgt_sos(self) -> None:
        model = Seq2Seq(8, 8, 4, 8, 2, 1, 0, 1, 2, dropout=0.0, max_len=8)
        model.encode = lambda src: (torch.zeros((1, 1, 4)), torch.zeros((1, 1), dtype=torch.bool))
        logits = [
            torch.tensor([[[0.0, 0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0]]]),
            torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0, 0.0]]]),
        ]
        model.decode = lambda *args: logits.pop(0)

        out = model.translate([1, 2], max_len=4, device=torch.device("cpu"), eos_idx=6)

        assert out == [4, 6]
