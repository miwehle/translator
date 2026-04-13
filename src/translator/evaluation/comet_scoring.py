from __future__ import annotations

import gc
import json
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from translator.inference import Translator


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    config: str | None = None
    split: str = "test"


def newstest_adapter(example: dict[str, Any]) -> dict[str, str]:
    translation = example["translation"]
    return {"src": str(translation["de"]), "ref": str(translation["en"])}


def translate(
    translator: Translator,
    test_dataset: DatasetSpec | Any,
    dataset_adapter: Callable[[dict[str, Any]], dict[str, str]],
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
            adapted_examples = [dataset_adapter(example) for example in examples]
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
        test_dataset: DatasetSpec = DatasetSpec("wmt20", "de-en", "test"),
        dataset_adapter: Callable[[dict[str, Any]], dict[str, str]] = newstest_adapter,
        translation_batch_size: int = 32,
        comet_batch_size: int = 8,
        output_path: str | Path | None = None,
    ) -> None:
        self.comet_model = comet_model
        self.test_dataset = test_dataset
        self.dataset_adapter = dataset_adapter
        self.translation_batch_size = translation_batch_size
        self.comet_batch_size = comet_batch_size
        self.output_path = output_path

    def score_checkpoint(self, checkpoint: str | Path) -> float:
        translator = Translator.from_checkpoint(checkpoint)
        uses_cuda = translator.device.type == "cuda"
        try:
            translations_path = translate(
                translator,
                self.test_dataset,
                self.dataset_adapter,
                output_path=self.output_path,
                batch_size=self.translation_batch_size,
            )
        finally:
            del translator
            gc.collect()
            if uses_cuda:
                torch.cuda.empty_cache()
        return comet_score(self.comet_model, translations_path, batch_size=self.comet_batch_size)


def _load_test_dataset(test_dataset: DatasetSpec | Any) -> Any:
    if isinstance(test_dataset, DatasetSpec):
        from datasets import load_dataset

        return load_dataset(test_dataset.path, test_dataset.config, split=test_dataset.split)
    return test_dataset


def _default_output_path() -> Path:
    return Path(".local_tmp") / "comet_translations.jsonl"


def _iter_batches(dataset: Any, batch_size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for example in dataset:
        batch.append(example)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch
