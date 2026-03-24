from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from .factory import TokenizerProtocol


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

    @classmethod
    def from_checkpoint_payload(
        cls, payload: dict[str, Any]
    ) -> "HuggingFaceTokenizerAdapter":
        files = payload.get("files")
        if not isinstance(files, dict) or not files:
            raise ValueError(
                "HF-Tokenizerdaten im Checkpoint sind ungueltig oder unvollstaendig."
            )
        tokenizer_name = payload.get("tokenizer_name")
        with TemporaryDirectory(prefix="hf_tok_load_") as tmp:
            tmp_path = Path(tmp)
            for name, blob in files.items():
                if not isinstance(name, str) or not isinstance(
                    blob, (bytes, bytearray)
                ):
                    raise ValueError(
                        "HF-Tokenizerdateien im Checkpoint haben ein "
                        "ungueltiges Format."
                    )
                (tmp_path / name).write_bytes(bytes(blob))

            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise ImportError(
                    "Checkpoint enthaelt HF-Tokenizerdaten, aber "
                    "'transformers' ist nicht installiert."
                ) from exc
            tokenizer = AutoTokenizer.from_pretrained(
                str(tmp_path), local_files_only=True
            )
        if isinstance(tokenizer_name, str):
            return cls(tokenizer, tokenizer_name=tokenizer_name)
        return cls(tokenizer)

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

    def to_checkpoint_payload(self) -> dict[str, Any]:
        with TemporaryDirectory(prefix="hf_tok_save_") as tmp:
            tmp_path = Path(tmp)
            self._tokenizer.save_pretrained(str(tmp_path))
            files: dict[str, bytes] = {}
            for path in tmp_path.iterdir():
                if path.is_file():
                    files[path.name] = path.read_bytes()
        return {
            "provider": "hf",
            "tokenizer_name": self._tokenizer_name,
            "files": files,
        }


def serialize_tokenizer(tokenizer: TokenizerProtocol) -> dict[str, Any]:
    if isinstance(tokenizer, HuggingFaceTokenizerAdapter):
        return tokenizer.to_checkpoint_payload()
    raise ValueError(
        "Tokenizer-Typ wird fuer Checkpoint-Speicherung nicht unterstuetzt. "
        "Erwartet: HuggingFaceTokenizerAdapter."
    )


def deserialize_tokenizer(payload: dict[str, Any]) -> TokenizerProtocol:
    provider = payload.get("provider")
    if provider == "hf":
        return HuggingFaceTokenizerAdapter.from_checkpoint_payload(payload)
    raise ValueError(f"Unbekannter Tokenizer-Provider im Checkpoint: {provider!r}.")
