import torch
import torch.nn as nn

from .factory import AttentionProtocol, create_attention


class EncoderBlock(nn.Module):
    """Transformer encoder block using Pre-Norm (LayerNorm before each sub-layer)."""

    def __init__(
        self, d_model: int, num_heads: int, ff_dim: int, dropout: float, attention: str
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn: AttentionProtocol = create_attention(
            attention, d_model, num_heads, dropout
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, src_key_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        y = self.norm1(x)
        attn_out, _ = self.self_attn(
            y,
            y,
            y,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop1(attn_out)

        y = self.norm2(x)
        x = x + self.drop2(self.ff(y))
        return x


class DecoderBlock(nn.Module):
    """Transformer decoder block using Pre-Norm (LayerNorm before each sub-layer)."""

    def __init__(
        self, d_model: int, num_heads: int, ff_dim: int, dropout: float, attention: str
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.self_attn: AttentionProtocol = create_attention(
            attention, d_model, num_heads, dropout
        )
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.cross_attn: AttentionProtocol = create_attention(
            attention, d_model, num_heads, dropout
        )
        self.drop2 = nn.Dropout(dropout)

        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, d_model),
        )
        self.drop3 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_key_padding_mask: torch.Tensor,
        src_key_padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        y = self.norm1(x)
        self_attn_out, _ = self.self_attn(
            y,
            y,
            y,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop1(self_attn_out)

        y = self.norm2(x)
        cross_attn_out, _ = self.cross_attn(
            y,
            memory,
            memory,
            key_padding_mask=src_key_padding_mask,
            need_weights=False,
        )
        x = x + self.drop2(cross_attn_out)

        y = self.norm3(x)
        x = x + self.drop3(self.ff(y))
        return x
