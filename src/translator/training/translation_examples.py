from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


class TranslationExamplesWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(
        self,
        step: int,
        epoch: int,
        loss: float,
        translations: Sequence[tuple[str, str]],
    ) -> None:
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(f"step={step} ep={epoch} loss={loss:.4f}\n")
            handle.write("---\n")
            for index, (source_text, translated_text) in enumerate(translations):
                if index > 0:
                    handle.write("\n")
                handle.write(f"src: {source_text}\n")
                handle.write(f"pred: {translated_text}\n")
            handle.write("\n")
