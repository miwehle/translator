from typing import Any

class HuggingFaceTokenizerAdapter:
    """Adapter exposing a minimal tokenizer contract compatible with this project."""

    def __init__(self, tokenizer: Any, tokenizer_name: str | None = None):
        self._tokenizer = tokenizer
        self._tokenizer_name = tokenizer_name

    @classmethod
    def from_pretrained(cls, model_name: str) -> "HuggingFaceTokenizerAdapter":
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Tokenizer 'hf' erfordert das Paket 'transformers'. "
                "Installiere es mit: python -m pip install transformers"
            ) from exc
        return cls(AutoTokenizer.from_pretrained(model_name), tokenizer_name=model_name)

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return list(self._tokenizer.encode(text, add_special_tokens=add_special_tokens))

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return str(self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens))

    def __call__(
        self,
        texts: str | list[str],
        padding: bool | str = False,
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        add_special_tokens: bool = True,
    ) -> dict[str, Any]:
        out = self._tokenizer(
            texts,
            padding=padding,
            truncation=truncation,
            max_length=max_length,
            return_tensors=return_tensors,
            add_special_tokens=add_special_tokens,
        )
        return dict(out)

    @property
    def vocab_size(self) -> int:
        size = getattr(self._tokenizer, "vocab_size", None)
        if isinstance(size, int):
            return size
        return len(self._tokenizer.get_vocab())

    @property
    def pad_token_id(self) -> int | None:
        return getattr(self._tokenizer, "pad_token_id", None)

    @property
    def bos_token_id(self) -> int | None:
        return getattr(self._tokenizer, "bos_token_id", None)

    @property
    def eos_token_id(self) -> int | None:
        return getattr(self._tokenizer, "eos_token_id", None)
