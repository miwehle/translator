from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn


@runtime_checkable
class AttentionProtocol(Protocol):
    """Shared attention contract matching PyTorch MHA.

    This keeps custom attention swappable with the PyTorch module.
    """

    def __call__(self, *args: Any, **kwargs: Any) -> tuple[torch.Tensor, torch.Tensor | None]: ...

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
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...

    def eval(self) -> "AttentionProtocol": ...


ATTENTION_CHOICES = ("torch", "simple_sdp")


def create_attention(attention: str, d_model: int, num_heads: int, dropout: float) -> AttentionProtocol:
    from .attention import SimpleMultiheadSDPAttention

    if attention == "torch":
        return nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
    if attention == "simple_sdp":
        return SimpleMultiheadSDPAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
    raise ValueError(f"Unknown attention={attention!r}. Allowed values: {ATTENTION_CHOICES}.")
