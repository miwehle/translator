from __future__ import annotations

import gc
import json
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from .config import DatasetConfig, MappingConfig

if TYPE_CHECKING:
    from translator.inference import Translator


def translate(
    translator: Translator,
    test_dataset: DatasetConfig | Any,
    mapping: MappingConfig,
    output_path: str | Path | None = None,
    batch_size: int = 32,
) -> Path:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    dataset = _load_test_dataset(test_dataset)
    destination = _default_output_path() if output_path is None else Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    with destination.open("w", encoding="utf-8") as handle:
        for examples in _iter_batches(dataset, batch_size):
            adapted_examples = [_map_example(example, mapping) for example in examples]
            hypotheses = translator.translate_many([example["src"] for example in adapted_examples])
            for adapted_example, hypothesis in zip(adapted_examples, hypotheses, strict=True):
                handle.write(
                    json.dumps(
                        {"src": adapted_example["src"], "hyp": hypothesis, "ref": adapted_example["ref"]},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    return destination


def comet_score(comet_model: str, translations_path: str | Path, batch_size: int = 8) -> float:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    from comet import download_model, load_from_checkpoint

    records = []
    with Path(translations_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records.append({"src": record["src"], "mt": record["hyp"], "ref": record["ref"]})

    model_path = download_model(comet_model)
    model = load_from_checkpoint(model_path)
    output = model.predict(records, batch_size=batch_size, gpus=1 if torch.cuda.is_available() else 0)
    return float(output.system_score)


class CometScorer:
    def __init__(
        self,
        comet_model: str = "Unbabel/wmt22-comet-da",
        test_dataset: DatasetConfig = DatasetConfig("wmt20", "de-en", "test"),
        mapping: MappingConfig = MappingConfig(src="translation.de", ref="translation.en"),
        translation_batch_size: int = 32,
        comet_batch_size: int = 8,
        output_path: str | Path | None = None,
    ) -> None:
        self.comet_model = comet_model
        self.test_dataset = test_dataset
        self.mapping = mapping
        self.translation_batch_size = translation_batch_size
        self.comet_batch_size = comet_batch_size
        self.output_path = output_path

    def score_checkpoint(self, checkpoint: str | Path) -> float:
        from translator.inference import Translator

        translator = Translator.from_checkpoint(checkpoint)
        uses_cuda = translator.device.type == "cuda"
        try:
            translations_path = translate(
                translator,
                self.test_dataset,
                self.mapping,
                output_path=self.output_path,
                batch_size=self.translation_batch_size,
            )
        finally:
            del translator
            gc.collect()
            if uses_cuda:
                torch.cuda.empty_cache()
        return comet_score(self.comet_model, translations_path, batch_size=self.comet_batch_size)


def _load_test_dataset(test_dataset: DatasetConfig | Any) -> Any:
    if isinstance(test_dataset, DatasetConfig):
        from datasets import load_dataset

        return load_dataset(
            path=test_dataset.path,
            name=test_dataset.name,
            split=test_dataset.split,
            data_files=test_dataset.data_files,
        )
    return test_dataset


def _default_output_path() -> Path:
    return Path(".local_tmp") / "comet_translations.jsonl"


def _map_example(example: dict[str, Any], mapping: MappingConfig) -> dict[str, str]:
    return {
        "src": _resolve_mapping_path(example, mapping.src),
        "ref": _resolve_mapping_path(example, mapping.ref),
    }


def _resolve_mapping_path(example: dict[str, Any], path: str) -> str:
    current: Any = example
    for part in path.split("."):
        current = current[part]
    return str(current)


def _iter_batches(dataset: Any, batch_size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for example in dataset:
        batch.append(example)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
