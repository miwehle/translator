from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenizerProtocol(Protocol):
    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]: ...

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str: ...

    @property
    def vocab_size(self) -> int: ...

    @property
    def pad_token_id(self) -> int | None: ...

    @property
    def bos_token_id(self) -> int | None: ...

    @property
    def eos_token_id(self) -> int | None: ...


TOKENIZER_CHOICES = ("custom", "hf")


def create_tokenizer(
    tokenizer: str, texts: list[str], hf_tokenizer_name: str
) -> TokenizerProtocol:
    from .tokenizer import HuggingFaceTokenizerAdapter, Tokenizer

    if tokenizer == "custom":
        return Tokenizer.build(texts)
    if tokenizer == "hf":
        return HuggingFaceTokenizerAdapter.from_pretrained(hf_tokenizer_name)
    raise ValueError(
        f"Unknown tokenizer={tokenizer!r}. Allowed values: {TOKENIZER_CHOICES}."
    )
