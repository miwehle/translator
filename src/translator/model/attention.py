import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMultiheadSDPAttention(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, dropout: float, batch_first: bool = True
    ):
        super().__init__()
        if not batch_first:
            raise ValueError(
                "SimpleMultiheadSDPAttention currently supports only batch_first=True"
            )
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = x.shape
        return x.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        bsz, _, seq_len, _ = x.shape
        return x.transpose(1, 2).contiguous().view(bsz, seq_len, self.embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        need_weights: bool = True,
        attn_mask: torch.Tensor | None = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Compute multi-head scaled dot-product attention.

        Mask convention: for boolean masks, True means "blocked"; for additive masks,
        values are added to attention logits (e.g. -inf blocks positions).
        """
        if is_causal:
            q_len = query.size(1)
            k_len = key.size(1)
            causal = torch.triu(
                torch.ones(q_len, k_len, device=query.device, dtype=torch.bool),
                diagonal=1,
            )
            if attn_mask is None:
                attn_mask = causal
            elif attn_mask.dtype == torch.bool:
                attn_mask = attn_mask | causal
            else:
                attn_mask = attn_mask + causal.to(attn_mask.dtype) * float("-inf")

        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                if attn_mask.dim() == 2:
                    scores = scores.masked_fill(
                        attn_mask.unsqueeze(0).unsqueeze(0), float("-inf")
                    )
                elif attn_mask.dim() == 3:
                    scores = scores.masked_fill(attn_mask.unsqueeze(1), float("-inf"))
                else:
                    raise ValueError(
                        "attn_mask with bool dtype must have shape "
                        "[Lq, Lk] or [B, Lq, Lk]"
                    )
            else:
                if attn_mask.dim() == 2:
                    scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
                elif attn_mask.dim() == 3:
                    scores = scores + attn_mask.unsqueeze(1)
                else:
                    raise ValueError(
                        "attn_mask must have shape [Lq, Lk] or [B, Lq, Lk]"
                    )

        if key_padding_mask is not None:
            scores = scores.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        attn_out = torch.matmul(attn_weights, v)
        out = self.out_proj(self._merge_heads(attn_out))

        if need_weights:
            if average_attn_weights:
                return out, attn_weights.mean(dim=1)
            return out, attn_weights
        return out, None
