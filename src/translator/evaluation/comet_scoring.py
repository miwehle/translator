from __future__ import annotations

import gc
import json
import logging
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from lab_infrastructure.logging import get_logger

from .config import DatasetConfig, MappingConfig

if TYPE_CHECKING:
    from translator.inference import Translator


logger = logging.getLogger("translator.evaluation.comet_scoring")


def translate(
    translator: Translator,
    test_dataset: DatasetConfig | Any,
    mapping: MappingConfig,
    output_path: str | Path | None = None,
    batch_size: int = 32,
) -> Path:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    get_logger("translator", stream=True)
    dataset = _load_test_dataset(test_dataset)
    destination = _default_output_path() if output_path is None else Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    total_examples = len(dataset) if hasattr(dataset, "__len__") else None
    total_batches = (
        None if total_examples is None else max(1, (total_examples + batch_size - 1) // batch_size)
    )
    logger.info(
        "Start translation examples=%s batch_size=%s output_path=%s",
        total_examples if total_examples is not None else "-",
        batch_size,
        destination,
    )

    with destination.open("w", encoding="utf-8") as handle:
        for batch_index, examples in enumerate(_iter_batches(dataset, batch_size), start=1):
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
            if _should_log_progress(batch_index, total_batches):
                translated_examples = batch_index * batch_size if total_examples is None else min(
                    total_examples, batch_index * batch_size
                )
                logger.info(
                    "Translation progress batches=%s/%s examples=%s/%s",
                    batch_index,
                    total_batches if total_batches is not None else "-",
                    translated_examples,
                    total_examples if total_examples is not None else "-",
                )

    logger.info("Finished translation output_path=%s", destination)

    return destination


def comet_score(comet_model: str, translations_path: str | Path, batch_size: int = 8) -> float:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    get_logger("translator", stream=True)
    from comet import download_model, load_from_checkpoint

    records = []
    with Path(translations_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            records.append({"src": record["src"], "mt": record["hyp"], "ref": record["ref"]})

    logger.info(
        "Load COMET model model=%s records=%s batch_size=%s", comet_model, len(records), batch_size
    )
    model_path = download_model(comet_model)
    model = load_from_checkpoint(model_path)
    logger.info("Run COMET prediction model=%s records=%s", comet_model, len(records))
    output = model.predict(records, batch_size=batch_size, gpus=1 if torch.cuda.is_available() else 0)
    logger.info("Finished COMET prediction model=%s score=%.6f", comet_model, float(output.system_score))
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


def _should_log_progress(batch_index: int, total_batches: int | None) -> bool:
    if batch_index == 1:
        return True
    if total_batches is None:
        return batch_index % 10 == 0
    if batch_index == total_batches:
        return True
    return batch_index % max(1, total_batches // 10) == 0
