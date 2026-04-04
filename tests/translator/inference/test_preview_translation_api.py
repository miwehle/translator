from __future__ import annotations

import pytest
import torch

from translator.inference import Translator, create_translation_preview_fn


class _FakeModel:
    pass


def test_create_translation_preview_fn_returns_none_without_schedule() -> None:
    preview_fn = create_translation_preview_fn(
        None,
        ["Hallo Welt"],
        "dummy-tokenizer",
        8,
        _FakeModel(),
        torch.device("cpu"),
    )

    assert preview_fn is None


def test_create_translation_preview_fn_raises_without_tokenizer_model_name() -> None:
    with pytest.raises(ValueError, match="dataset tokenizer_model_name"):
        create_translation_preview_fn(
            10,
            ["Hallo Welt"],
            None,
            8,
            _FakeModel(),
            torch.device("cpu"),
        )


def test_create_translation_preview_fn_builds_callable_from_public_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_tokenizer = object()
    fake_model = _FakeModel()
    device = torch.device("cpu")
    calls: dict[str, object] = {}

    def fake_load_preview_tokenizer(tokenizer_model_name: str) -> object:
        calls["tokenizer_model_name"] = tokenizer_model_name
        return fake_tokenizer

    def fake_translate_examples(
        model: object,
        tokenizer: object,
        texts: list[str],
        device_arg: torch.device,
        *,
        tgt_bos_id: int | None,
    ) -> list[tuple[str, str]]:
        calls["model"] = model
        calls["tokenizer"] = tokenizer
        calls["texts"] = texts
        calls["device"] = device_arg
        calls["tgt_bos_id"] = tgt_bos_id
        return [("Hallo Welt", "Hello world")]

    monkeypatch.setattr(
        "translator.inference.preview_translation._load_preview_tokenizer",
        fake_load_preview_tokenizer,
    )
    monkeypatch.setattr(
        "translator.inference.preview_translation.translate_examples",
        fake_translate_examples,
    )

    preview_fn = create_translation_preview_fn(
        10,
        ["Hallo Welt"],
        "dummy-tokenizer",
        8,
        fake_model,
        device,
    )

    assert preview_fn is not None
    assert preview_fn() == [("Hallo Welt", "Hello world")]
    assert calls == {
        "tokenizer_model_name": "dummy-tokenizer",
        "model": fake_model,
        "tokenizer": fake_tokenizer,
        "texts": ["Hallo Welt"],
        "device": device,
        "tgt_bos_id": 8,
    }


def test_translate_examples_delegates_to_translator() -> None:
    model = _FakeModel()
    tokenizer = object()
    device = torch.device("cpu")
    calls: dict[str, object] = {}

    class _FakeTranslator:
        def __init__(
            self,
            model_arg: object,
            tokenizer_arg: object,
            device_arg: torch.device,
            tgt_bos_id_arg: int | None,
        ) -> None:
            calls["model"] = model_arg
            calls["tokenizer"] = tokenizer_arg
            calls["device"] = device_arg
            calls["tgt_bos_id"] = tgt_bos_id_arg

        def translate_many(self, texts: list[str]) -> list[str]:
            calls["texts"] = texts
            return ["Hello world"]

    original = Translator
    try:
        import translator.inference.preview_translation as preview_translation

        preview_translation.Translator = _FakeTranslator
        assert preview_translation.translate_examples(
            model, tokenizer, ["Hallo Welt"], device, tgt_bos_id=8
        ) == [("Hallo Welt", "Hello world")]
    finally:
        import translator.inference.preview_translation as preview_translation

        preview_translation.Translator = original

    assert calls == {
        "model": model,
        "tokenizer": tokenizer,
        "device": device,
        "tgt_bos_id": 8,
        "texts": ["Hallo Welt"],
    }
