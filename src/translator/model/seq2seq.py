import math

import torch
import torch.nn as nn

from .blocks import DecoderBlock, EncoderBlock
from .factory import ATTENTION_CHOICES


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.pe: torch.Tensor
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class Seq2Seq(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        ff_dim: int,
        num_heads: int,
        num_layers: int,
        src_pad_idx: int,
        tgt_pad_idx: int,
        tgt_sos_idx: int,
        dropout: float = 0.1,
        max_len: int = 1024,
        attention: str = "torch",
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if attention not in ATTENTION_CHOICES:
            raise ValueError(
                f"Unknown attention={attention!r}. Allowed values: {ATTENTION_CHOICES}."
            )

        self.src_pad_idx = src_pad_idx
        self.tgt_pad_idx = tgt_pad_idx
        self.tgt_sos_idx = tgt_sos_idx
        self.d_model = d_model

        self.src_embed = nn.Embedding(src_vocab_size, d_model, padding_idx=src_pad_idx)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model, padding_idx=tgt_pad_idx)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.embed_dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList(
            [
                EncoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    attention=attention,
                )
                for _ in range(num_layers)
            ]
        )
        self.decoder_layers = nn.ModuleList(
            [
                DecoderBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                    attention=attention,
                )
                for _ in range(num_layers)
            ]
        )

        self.enc_final_norm = nn.LayerNorm(d_model)
        self.dec_final_norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, tgt_vocab_size)

    def _causal_mask(self, length: int, device: torch.device) -> torch.Tensor:
        return torch.triu(
            torch.ones(length, length, device=device, dtype=torch.bool), diagonal=1
        )

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        src_key_padding_mask = src == self.src_pad_idx
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = self.embed_dropout(self.pos_enc(x))
        for layer in self.encoder_layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        x = self.enc_final_norm(x)
        return x, src_key_padding_mask

    def decode(
        self,
        tgt_in: torch.Tensor,
        memory: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        tgt_key_padding_mask = tgt_in == self.tgt_pad_idx
        tgt_mask = self._causal_mask(tgt_in.size(1), tgt_in.device)

        x = self.tgt_embed(tgt_in) * math.sqrt(self.d_model)
        x = self.embed_dropout(self.pos_enc(x))
        for layer in self.decoder_layers:
            x = layer(
                x,
                memory=memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                src_key_padding_mask=src_key_padding_mask,
            )
        x = self.dec_final_norm(x)
        return self.out_proj(x)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        memory, src_key_padding_mask = self.encode(src)
        # Teacher forcing input: decoder sees target tokens up to t-1
        # to predict token t.
        tgt_in = tgt[:, :-1]
        return self.decode(tgt_in, memory, src_key_padding_mask)

    @torch.no_grad()
    def translate(
        self, src_ids: list[int], max_len: int, device: torch.device, eos_idx: int
    ) -> list[int]:
        """Simple inference: greedy argmax decoding only.

        This intentionally does not implement beam search or sampling.
        Returns token IDs that are ready for tokenizer.decode().
        """
        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        memory, src_key_padding_mask = self.encode(src)

        out_ids: list[int] = [self.tgt_sos_idx]
        for _ in range(max_len):
            tgt_in = torch.tensor([out_ids], dtype=torch.long, device=device)
            logits = self.decode(tgt_in, memory, src_key_padding_mask)
            next_token = int(logits[:, -1, :].argmax(dim=-1).item())
            out_ids.append(next_token)
            if next_token == eos_idx:
                break
        return out_ids[1:]

    @torch.no_grad()
    def translate_beam(
        self, src_ids: list[int], max_len: int, device: torch.device, eos_idx: int,
        beam_width: int = 4, length_penalty: float = 0.6
    ) -> list[int]:
        if beam_width < 1:
            raise ValueError("beam_width must be at least 1.")
        if length_penalty < 0:
            raise ValueError("length_penalty must be non-negative.")

        def normalized_score(token_ids: list[int], score: float) -> float:
            length = max(len(token_ids) - 1, 1)
            penalty = ((5 + length) / 6) ** length_penalty
            return score / penalty

        src = torch.tensor([src_ids], dtype=torch.long, device=device)
        memory, src_key_padding_mask = self.encode(src)
        beams: list[tuple[list[int], float]] = [([self.tgt_sos_idx], 0.0)]

        for _ in range(max_len):
            candidates: list[tuple[list[int], float]] = []
            all_finished = True

            for token_ids, score in beams:
                if token_ids[-1] == eos_idx:
                    candidates.append((token_ids, score))
                    continue

                all_finished = False
                tgt_in = torch.tensor([token_ids], dtype=torch.long, device=device)
                logits = self.decode(tgt_in, memory, src_key_padding_mask)
                log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)
                top_log_probs, top_token_ids = torch.topk(log_probs, beam_width, dim=-1)

                for next_log_prob, next_token_id in zip(
                    top_log_probs[0].tolist(), top_token_ids[0].tolist(), strict=True
                ):
                    candidates.append((token_ids + [int(next_token_id)], score + next_log_prob))

            if all_finished:
                break

            beams = sorted(
                candidates,
                key=lambda candidate: normalized_score(candidate[0], candidate[1]),
                reverse=True,
            )[:beam_width]

        best_token_ids, _ = max(
            beams, key=lambda candidate: normalized_score(candidate[0], candidate[1])
        )
        return best_token_ids[1:]
