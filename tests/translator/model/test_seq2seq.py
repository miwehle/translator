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

    def test_translate_beam_returns_best_decode_ready_ids_without_initial_tgt_sos(self) -> None:
        model = Seq2Seq(8, 8, 4, 8, 2, 1, 0, 1, 2, dropout=0.0, max_len=8)
        model.encode = lambda src: (torch.zeros((1, 1, 4)), torch.zeros((1, 1), dtype=torch.bool))
        decode_outputs = {
            (2,): torch.tensor([[[0.0, 0.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0]]]),
            (2, 4): torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0]]]),
            (2, 5): torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 4.0]]]),
            (2, 5, 7): torch.tensor([[[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0, 0.0]]]),
        }

        def fake_decode(tgt_in: torch.Tensor, *_args) -> torch.Tensor:
            return decode_outputs[tuple(tgt_in[0].tolist())]

        model.decode = fake_decode

        out = model.translate_beam([1, 2], max_len=4, device=torch.device("cpu"), eos_idx=6, beam_width=2)

        assert out == [5, 7, 6]

    def test_translate_beam_rejects_invalid_beam_width(self) -> None:
        model = Seq2Seq(8, 8, 4, 8, 2, 1, 0, 1, 2, dropout=0.0, max_len=8)

        try:
            model.translate_beam([1, 2], max_len=4, device=torch.device("cpu"), eos_idx=6, beam_width=0)
        except ValueError as exc:
            assert str(exc) == "beam_width must be at least 1."
        else:
            raise AssertionError("Expected ValueError for beam_width < 1.")
