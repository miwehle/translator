from __future__ import annotations

import runpy
import sys
from pathlib import Path
from uuid import uuid4

import pytest
import yaml


def test_script_train_loads_yaml_and_calls_api(monkeypatch):
    tmp_path = Path(__file__).resolve().parents[1] / ".local_tmp" / "tests" / uuid4().hex
    tmp_path.mkdir(parents=True, exist_ok=True)
    config_path = tmp_path / "train_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "resume_run": "older-run",
                "model_config": {"d_model": 128},
                "train_config": {"dataset": "demo-dataset", "run_name": "exp1"},
                "data_loader_config": {"batch_size": 16, "shuffle": False},
            }
        ),
        encoding="utf-8",
    )

    calls: list[dict[str, object]] = []

    def fake_train(config):
        calls.append({"config": config})

    monkeypatch.setattr(sys, "argv", ["train.py", str(config_path)])
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "src"))
    import translator as api

    monkeypatch.setattr(api, "train", fake_train)

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "train.py"), run_name="__main__")

    assert excinfo.value.code == 0
    assert calls == [
        {
            "config": api.TrainRunConfig(
                train_config=api.TrainConfig(dataset="demo-dataset", run_name="exp1"),
                data_loader_config=api.DataLoaderConfig(batch_size=16, shuffle=False),
                model_config=api.ModelConfig(d_model=128),
                resume_run="older-run",
            )
        }
    ]
