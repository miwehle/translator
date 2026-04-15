from __future__ import annotations

import json
from pathlib import Path

import torch

from translator.evaluation import (
    CometScorer,
    DatasetConfig,
    MappingConfig,
    comet_score,
    translate,
)
from translator.evaluation.comet_scoring import _load_test_dataset


class _FakeTranslator:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.calls: list[list[str]] = []

    def translate_many(self, texts: list[str]) -> list[str]:
        self.calls.append(texts)
        return [text.upper() for text in texts]


class _FakeCometOutput:
    def __init__(self, system_score: float) -> None:
        self.system_score = system_score


class _FakeCometModel:
    def __init__(self, system_score: float) -> None:
        self.system_score = system_score
        self.calls: list[tuple[list[dict[str, str]], int, int]] = []

    def predict(self, data: list[dict[str, str]], batch_size: int, gpus: int) -> _FakeCometOutput:
        self.calls.append((data, batch_size, gpus))
        return _FakeCometOutput(self.system_score)


class TestTranslate:
    def test_writes_jsonl_with_src_hyp_ref(self, tmp_path: Path) -> None:
        translator = _FakeTranslator()
        output_path = tmp_path / "translations.jsonl"
        mapping = MappingConfig(src="translation.de", ref="translation.en")
        dataset = [
            {"translation": {"de": "eins", "en": "one"}},
            {"translation": {"de": "zwei", "en": "two"}},
        ]

        written_path = translate(translator, dataset, mapping, output_path=output_path, batch_size=2)

        assert written_path == output_path
        assert translator.calls == [["eins", "zwei"]]
        assert output_path.read_text(encoding="utf-8").splitlines() == [
            json.dumps({"src": "eins", "hyp": "EINS", "ref": "one"}, ensure_ascii=False),
            json.dumps({"src": "zwei", "hyp": "ZWEI", "ref": "two"}, ensure_ascii=False),
        ]

    def test_uses_default_output_path_when_none(self, tmp_path: Path, monkeypatch) -> None:
        translator = _FakeTranslator()
        mapping = MappingConfig(src="translation.de", ref="translation.en")
        dataset = [{"translation": {"de": "eins", "en": "one"}}]
        default_path = tmp_path / ".local_tmp" / "comet_translations.jsonl"
        monkeypatch.setattr("translator.evaluation.comet_scoring._default_output_path", lambda: default_path)

        written_path = translate(translator, dataset, mapping, output_path=None, batch_size=1)

        assert written_path == default_path
        assert default_path.is_file()


class TestCometScore:
    def test_returns_system_score(self, tmp_path: Path, monkeypatch) -> None:
        translations_path = tmp_path / "translations.jsonl"
        translations_path.write_text(
            json.dumps({"src": "eins", "hyp": "one", "ref": "one"}) + "\n",
            encoding="utf-8",
        )
        fake_model = _FakeCometModel(0.75)
        monkeypatch.setitem(
            __import__("sys").modules,
            "comet",
            type(
                "CometModule",
                (),
                {
                    "download_model": staticmethod(lambda model: f"downloaded::{model}"),
                    "load_from_checkpoint": staticmethod(lambda path: fake_model),
                },
            ),
        )

        score = comet_score("Unbabel/wmt22-comet-da", translations_path, batch_size=4)

        assert score == 0.75
        assert fake_model.calls == [([{"src": "eins", "mt": "one", "ref": "one"}], 4, 0)]


class TestCometScorer:
    def test_score_checkpoint_releases_translator_before_comet(self, monkeypatch) -> None:
        cleanup_steps: list[str] = []
        translator = _FakeTranslator()

        class _TranslatorFactory:
            @staticmethod
            def from_checkpoint(checkpoint: str | Path) -> _FakeTranslator:
                del checkpoint
                return translator

        def fake_translate(*args, **kwargs) -> Path:
            del args, kwargs
            return Path("translations.jsonl")

        def fake_collect() -> int:
            cleanup_steps.append("gc.collect")
            return 0

        def fake_comet_score(model: str, path: str | Path, batch_size: int) -> float:
            del model, path, batch_size
            cleanup_steps.append("comet_score")
            return 0.42

        monkeypatch.setattr("translator.inference.Translator", _TranslatorFactory)
        monkeypatch.setattr("translator.evaluation.comet_scoring.translate", fake_translate)
        monkeypatch.setattr("translator.evaluation.comet_scoring.gc.collect", fake_collect)
        monkeypatch.setattr("translator.evaluation.comet_scoring.comet_score", fake_comet_score)

        scorer = CometScorer(
            test_dataset=DatasetConfig("wmt20"), mapping=MappingConfig("translation.de", "translation.en")
        )

        score = scorer.score_checkpoint("checkpoint.pt")

        assert score == 0.42
        assert cleanup_steps == ["gc.collect", "comet_score"]


class TestLoadTestDataset:
    def test_passes_name_and_data_files_to_load_dataset(self, monkeypatch) -> None:
        captured: dict[str, object] = {}

        def fake_load_dataset(
            *, path: str, name: str | None, split: str, data_files: str | None
        ) -> list[object]:
            captured["path"] = path
            captured["name"] = name
            captured["split"] = split
            captured["data_files"] = data_files
            return []

        monkeypatch.setitem(
            __import__("sys").modules,
            "datasets",
            type("DatasetsModule", (), {"load_dataset": staticmethod(fake_load_dataset)}),
        )

        dataset = DatasetConfig("json", split="train", data_files="/content/drive/MyDrive/filtered.jsonl")
        assert _load_test_dataset(dataset) == []
        assert captured == {
            "path": "json",
            "name": None,
            "split": "train",
            "data_files": "/content/drive/MyDrive/filtered.jsonl",
        }


class TestTranslateMapping:
    def test_maps_src_and_ref_from_mapping_paths(self, tmp_path: Path) -> None:
        translator = _FakeTranslator()
        output_path = tmp_path / "translations.jsonl"

        translate(
            translator,
            [{"translation": {"de": "Hallo", "en": "Hello"}}],
            MappingConfig(src="translation.de", ref="translation.en"),
            output_path=output_path,
            batch_size=1,
        )

        assert output_path.read_text(encoding="utf-8").splitlines() == [
            json.dumps({"src": "Hallo", "hyp": "HALLO", "ref": "Hello"}, ensure_ascii=False)
        ]
