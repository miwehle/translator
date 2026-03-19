from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import checkpoint_register as cr

from translator.data_prod import DatasetMetadata, load_arrow_records
from translator.train_prod import (
    Example,
    Trainer,
    TrainerConfig,
    build_model,
    check_dataset,
)


@dataclass(frozen=True)
class TrainingRunConfig:
    dataset_path: str
    runs_dir: str
    run_name: str
    run_preflight_check: bool = True


CONFIG = TrainingRunConfig(
    dataset_path="/content/drive/MyDrive/nmt_lab/artifacts/datasets/europarl_de-en_train_10000",
    runs_dir="/content/drive/MyDrive/nmt_lab/artifacts/training_runs",
    run_name="run1",
)


def create_run_dir(runs_dir: Path, run_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = runs_dir / f"{timestamp}_{run_name}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def get_git_commit_hash(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def write_training_run_config(
    run_dir: Path,
    config: TrainingRunConfig,
    *,
    build_commit: str,
) -> Path:
    config_path = run_dir / "config.json"
    payload = asdict(config)
    payload["build_commit"] = build_commit
    config_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return config_path


def main(config: TrainingRunConfig = CONFIG) -> dict[str, object]:
    """
    load dataset
    create model
    create trainer
    trainer, train model on dataset
    save model
    """
    dataset_path = Path(config.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    run_dir = create_run_dir(Path(config.runs_dir), config.run_name)
    git_commit = get_git_commit_hash(REPO_ROOT)
    config_path = write_training_run_config(
        run_dir,
        config,
        build_commit=git_commit,
    )

    ds = cast(list[Example], load_arrow_records(dataset_path))
    if config.run_preflight_check:
        dataset_info = check_dataset(ds)
    else:
        metadata = DatasetMetadata.from_file(dataset_path / "dataset_manifest.yaml")
        dataset_info = {
            "id_field": metadata.id_field,
            "src_field": metadata.src_field,
            "tgt_field": metadata.tgt_field,
            "src_pad_idx": metadata.src_pad_id,
            "tgt_pad_idx": metadata.tgt_pad_id,
            "tgt_sos_idx": metadata.tgt_bos_id,
            "src_vocab_size": metadata.src_vocab_size,
            "tgt_vocab_size": metadata.tgt_vocab_size,
        }
    seed = 42
    device = None

    model = build_model(
        src_vocab_size=dataset_info["src_vocab_size"],
        tgt_vocab_size=dataset_info["tgt_vocab_size"],
        src_pad_idx=dataset_info["src_pad_idx"],
        tgt_pad_idx=dataset_info["tgt_pad_idx"],
        tgt_sos_idx=dataset_info["tgt_sos_idx"],
        device=device,
        seed=seed,
    )

    trainer_config = TrainerConfig(
        id_field=dataset_info["id_field"],
        src_field=dataset_info["src_field"],
        tgt_field=dataset_info["tgt_field"],
        device=device,
        seed=seed,
    )

    summary = Trainer(model, trainer_config).train(
        ds,
        epochs=1,
        num_workers=1,
        log_every=50,
        checkpoint_path=run_dir / "model.pt",
        summary_path=run_dir / "summary.json",
    )
    register_path = Path(config.runs_dir) / "checkpoint_register.csv"
    cr.insert(
        register_path=register_path,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        input_ckpt="",
        dataset_path=str(dataset_path),
        git_commit=git_commit,
        output_ckpt=str(run_dir / "model.pt"),
    )
    summary["config_path"] = str(config_path)
    summary["build_commit"] = git_commit
    summary["checkpoint_register_path"] = str(register_path)
    summary["run_dir"] = str(run_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


if __name__ == "__main__":
    main()
