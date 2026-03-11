import torch
import torch.nn as nn

from translator.model.attention import SimpleMultiheadSDPAttention
from translator.model import Seq2Seq
from translator.model.factory import (
    ATTENTION_CHOICES,
    AttentionProtocol,
    create_attention,
)


def _dummy_batch(batch_size: int = 2, src_len: int = 5, tgt_len: int = 6):
    src = torch.tensor(
        [
            [2, 7, 8, 0, 0],
            [2, 5, 6, 9, 0],
        ],
        dtype=torch.long,
    )[:batch_size, :src_len]
    tgt = torch.tensor(
        [
            [2, 4, 3, 1, 0, 0],
            [2, 6, 7, 8, 1, 0],
        ],
        dtype=torch.long,
    )[:batch_size, :tgt_len]
    return src, tgt


def _make_model(attention: str):
    return Seq2Seq(
        src_vocab_size=32,
        tgt_vocab_size=40,
        d_model=16,
        ff_dim=32,
        num_heads=4,
        num_layers=1,
        src_pad_idx=0,
        tgt_pad_idx=0,
        tgt_sos_idx=2,
        dropout=0.0,
        max_len=32,
        attention=attention,
    )


def test_attention_choices_create_protocol_compatible_modules():
    for attention in ATTENTION_CHOICES:
        attn = create_attention(attention, 16, 4, 0.0)
        assert isinstance(attn, nn.Module)
        assert isinstance(attn, AttentionProtocol)


def _copy_torch_mha_weights_to_simple(
    torch_attn: nn.MultiheadAttention, simple_attn: SimpleMultiheadSDPAttention
) -> None:
    d_model = torch_attn.embed_dim
    with torch.no_grad():
        simple_attn.q_proj.weight.copy_(torch_attn.in_proj_weight[:d_model, :])
        simple_attn.k_proj.weight.copy_(
            torch_attn.in_proj_weight[d_model : 2 * d_model, :]
        )
        simple_attn.v_proj.weight.copy_(torch_attn.in_proj_weight[2 * d_model :, :])
        simple_attn.q_proj.bias.copy_(torch_attn.in_proj_bias[:d_model])
        simple_attn.k_proj.bias.copy_(torch_attn.in_proj_bias[d_model : 2 * d_model])
        simple_attn.v_proj.bias.copy_(torch_attn.in_proj_bias[2 * d_model :])
        simple_attn.out_proj.weight.copy_(torch_attn.out_proj.weight)
        simple_attn.out_proj.bias.copy_(torch_attn.out_proj.bias)


def test_seq2seq_forward_works_with_torch_attention_factory():
    model = _make_model("torch")
    src, tgt = _dummy_batch()
    logits = model(src, tgt)
    assert logits.shape == (src.size(0), tgt.size(1) - 1, 40)


def test_seq2seq_forward_works_with_simple_sdp_attention_factory():
    model = _make_model("simple_sdp")
    src, tgt = _dummy_batch()
    logits = model(src, tgt)
    assert logits.shape == (src.size(0), tgt.size(1) - 1, 40)


def test_simple_sdp_attention_applies_masks_and_returns_expected_shapes():
    attn = create_attention("simple_sdp", 16, 4, 0.0)

    query = torch.randn(2, 4, 16)
    key = torch.randn(2, 5, 16)
    value = torch.randn(2, 5, 16)
    attn_mask = torch.zeros(4, 5, dtype=torch.bool)
    attn_mask[:, -1] = True
    key_padding_mask = torch.tensor(
        [
            [False, False, False, False, True],
            [False, False, False, True, True],
        ],
        dtype=torch.bool,
    )

    out, weights = attn(
        query,
        key,
        value,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
        need_weights=True,
    )
    assert out.shape == (2, 4, 16)
    assert weights is not None
    assert weights.shape == (2, 4, 5)


def test_invalid_attention_choice_raises_clear_error():
    try:
        _make_model("invalid_attention")
        assert False, "expected ValueError"
    except ValueError:
        assert True


def test_simple_sdp_is_causal_matches_explicit_causal_mask():
    torch.manual_seed(0)
    attn = create_attention("simple_sdp", 16, 4, 0.0)
    assert isinstance(attn, SimpleMultiheadSDPAttention)
    query = torch.randn(2, 5, 16)
    key = torch.randn(2, 5, 16)
    value = torch.randn(2, 5, 16)
    explicit_causal = torch.triu(torch.ones(5, 5, dtype=torch.bool), diagonal=1)

    out_is_causal, w_is_causal = attn(
        query, key, value, is_causal=True, need_weights=True
    )
    out_explicit, w_explicit = attn(
        query, key, value, attn_mask=explicit_causal, need_weights=True
    )

    assert torch.allclose(out_is_causal, out_explicit, atol=1e-6, rtol=1e-5)
    assert w_is_causal is not None and w_explicit is not None
    assert torch.allclose(w_is_causal, w_explicit, atol=1e-6, rtol=1e-5)


def test_simple_sdp_matches_torch_mha_protocol_behavior():
    torch_attn = create_attention("torch", 16, 4, 0.0)
    simple_attn = create_attention("simple_sdp", 16, 4, 0.0)

    query = torch.randn(2, 4, 16)
    key = torch.randn(2, 5, 16)
    value = torch.randn(2, 5, 16)
    attn_mask = torch.zeros(4, 5, dtype=torch.bool)
    attn_mask[:, -1] = True
    key_padding_mask = torch.tensor(
        [[False, False, False, False, True], [False, False, False, True, True]],
        dtype=torch.bool,
    )

    out_torch, w_torch = torch_attn(
        query,
        key,
        value,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
        need_weights=True,
        average_attn_weights=True,
    )
    out_simple, w_simple = simple_attn(
        query,
        key,
        value,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
        need_weights=True,
        average_attn_weights=True,
    )

    assert out_torch.shape == out_simple.shape == (2, 4, 16)
    assert w_simple is not None and w_torch is not None
    assert w_torch.shape == w_simple.shape == (2, 4, 5)
    assert torch.isfinite(out_torch).all()
    assert torch.isfinite(out_simple).all()
    assert torch.isfinite(w_torch).all()
    assert torch.isfinite(w_simple).all()

    # Last key is masked by attn_mask for all queries.
    assert torch.all(w_torch[:, :, -1] == 0)
    assert torch.all(w_simple[:, :, -1] == 0)


def test_simple_sdp_matches_torch_mha_with_same_weights_backdoor():
    # Intentional backdoor test: we align internal weights to verify strict
    # numerical parity between both implementations. Protocol-only blackbox
    # checks cannot prove this level of equivalence.
    torch.manual_seed(1)
    torch_attn = create_attention("torch", 16, 4, 0.0)
    simple_attn = create_attention("simple_sdp", 16, 4, 0.0)
    assert isinstance(torch_attn, nn.MultiheadAttention)
    assert isinstance(simple_attn, SimpleMultiheadSDPAttention)

    _copy_torch_mha_weights_to_simple(torch_attn, simple_attn)
    torch_attn.eval()
    simple_attn.eval()

    query = torch.randn(2, 4, 16)
    key = torch.randn(2, 5, 16)
    value = torch.randn(2, 5, 16)
    attn_mask = torch.zeros(4, 5, dtype=torch.bool)
    attn_mask[:, -1] = True
    key_padding_mask = torch.tensor(
        [[False, False, False, False, True], [False, False, False, True, True]],
        dtype=torch.bool,
    )

    out_torch, w_torch = torch_attn(
        query,
        key,
        value,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
        need_weights=True,
        average_attn_weights=True,
    )
    out_simple, w_simple = simple_attn(
        query,
        key,
        value,
        attn_mask=attn_mask,
        key_padding_mask=key_padding_mask,
        need_weights=True,
        average_attn_weights=True,
    )

    assert torch.allclose(out_simple, out_torch, atol=1e-5, rtol=1e-4)
    assert w_simple is not None and w_torch is not None
    assert torch.allclose(w_simple, w_torch, atol=1e-5, rtol=1e-4)


def test_simple_sdp_invalid_attn_mask_shape_raises_value_error():
    attn = create_attention("simple_sdp", 16, 4, 0.0)
    assert isinstance(attn, SimpleMultiheadSDPAttention)
    query = torch.randn(2, 4, 16)
    key = torch.randn(2, 5, 16)
    value = torch.randn(2, 5, 16)

    bad_bool_mask = torch.zeros(2, 1, 4, 5, dtype=torch.bool)
    bad_additive_mask = torch.zeros(2, 1, 4, 5)

    try:
        attn(query, key, value, attn_mask=bad_bool_mask)
        assert False, "expected ValueError for bool attn_mask shape"
    except ValueError:
        assert True

    try:
        attn(query, key, value, attn_mask=bad_additive_mask)
        assert False, "expected ValueError for additive attn_mask shape"
    except ValueError:
        assert True
