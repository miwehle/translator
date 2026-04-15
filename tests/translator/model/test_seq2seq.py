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

    def test_translate_beam_clamps_decode_length_to_model_max_len(self) -> None:
        model = Seq2Seq(8, 8, 4, 8, 2, 1, 0, 1, 2, dropout=0.0, max_len=4)
        model.encode = lambda src: (torch.zeros((1, 1, 4)), torch.zeros((1, 1), dtype=torch.bool))
        observed_lengths: list[int] = []

        def fake_decode(tgt_in: torch.Tensor, *_args) -> torch.Tensor:
            observed_lengths.append(int(tgt_in.size(1)))
            logits = torch.full((1, tgt_in.size(1), 8), -1e9, dtype=torch.float32)
            logits[:, -1, 3] = 0.0
            return logits

        model.decode = fake_decode

        model.translate_beam([1, 2], max_len=10, device=torch.device("cpu"), eos_idx=7, beam_width=1)

        assert observed_lengths == [1, 2, 3]

    def test_translate_batch_returns_decode_ready_ids_without_initial_tgt_sos(self) -> None:
        model = Seq2Seq(8, 8, 4, 8, 2, 1, 0, 1, 2, dropout=0.0, max_len=8)
        model.encode = lambda src: (
            torch.zeros((src.size(0), src.size(1), 4)),
            torch.zeros((src.size(0), src.size(1)), dtype=torch.bool),
        )
        step1 = torch.full((2, 1, 8), -1e9)
        step1[0, 0, 4] = 5.0
        step1[1, 0, 3] = 3.0
        step2 = torch.full((2, 2, 8), -1e9)
        step2[0, 1, 6] = 6.0
        step2[1, 1, 5] = 7.0
        logits = [
            step1,
            step2,
        ]
        model.decode = lambda *args: logits.pop(0)

        out = model.translate_batch([[1, 2], [3]], max_len=2, device=torch.device("cpu"), eos_idx=6)

        assert out == [[4, 6], [3, 5]]

    def test_translate_beam_batch_returns_best_decode_ready_ids_without_initial_tgt_sos(self) -> None:
        model = Seq2Seq(8, 8, 4, 8, 2, 1, 0, 1, 2, dropout=0.0, max_len=8)
        model.encode = lambda src: (
            torch.zeros((src.size(0), src.size(1), 4)),
            torch.zeros((src.size(0), src.size(1)), dtype=torch.bool),
        )
        first_step = torch.full((2, 1, 8), -1e9)
        first_step[0, 0, 5] = 5.0
        first_step[0, 0, 4] = 4.0
        first_step[1, 0, 4] = 6.0
        first_step[1, 0, 3] = 3.0
        second_step = torch.full((4, 2, 8), -1e9)
        second_step[0, 1, 7] = 4.0
        second_step[0, 1, 6] = 1.0
        second_step[1, 1, 6] = 5.0
        second_step[1, 1, 7] = 1.0
        second_step[2, 1, 6] = 5.0
        second_step[2, 1, 7] = 1.0
        second_step[3, 1, 7] = 5.0
        second_step[3, 1, 6] = 2.0
        third_step = torch.full((2, 3, 8), -1e9)
        third_step[0, 2, 6] = 5.0
        third_step[0, 2, 7] = 1.0
        third_step[1, 2, 6] = 2.0
        third_step[1, 2, 7] = 1.0
        decode_outputs = {
            ((2,), (2,)): first_step,
            ((2, 5), (2, 4), (2, 4), (2, 3)): second_step,
            ((2, 5, 7), (2, 3, 7)): third_step,
        }

        def fake_decode(tgt_in: torch.Tensor, *_args) -> torch.Tensor:
            sequences = tuple(tuple(row[: int((row != 1).sum())].tolist()) for row in tgt_in)
            return decode_outputs[sequences]

        model.decode = fake_decode

        out = model.translate_beam_batch(
            [[1, 2], [3, 4]], max_len=4, device=torch.device("cpu"), eos_idx=6, beam_width=2
        )

        assert out == [[5, 7, 6], [4, 6]]
