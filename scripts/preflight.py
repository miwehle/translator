from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from translator.data_prod import DatasetMetadata, load_arrow_records
from translator.train_prod import check_dataset


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/preflight.py <config-path>")
        return 1

    config_path = Path(sys.argv[1])
    try:
        with config_path.open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"Failed to load config: {exc}")
        return 1

    try:
        dataset_path = Path(cfg["dataset_path"])
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        manifest_path = dataset_path / "dataset_manifest.yaml"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Dataset manifest not found: {manifest_path}")

        examples = load_arrow_records(dataset_path)
        metadata = DatasetMetadata.from_file(manifest_path)
        extra_preflight_config = cfg.get("preflight_config") or {}
        result = check_dataset(
            examples,
            id_field=metadata.id_field,
            src_field=metadata.src_field,
            tgt_field=metadata.tgt_field,
            src_pad_idx=metadata.src_pad_id,
            tgt_pad_idx=metadata.tgt_pad_id,
            **extra_preflight_config,
        )
    except Exception as exc:
        print(f"Preflight check failed: {exc}")
        return 1

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
