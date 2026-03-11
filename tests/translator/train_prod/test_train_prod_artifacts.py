from __future__ import annotations

from pathlib import Path

from translator.train_prod import build_train_prod_config, run_train_prod

from tests.translator.train_prod.test_train_prod_loss_progress import (
    _create_valid_mapped_dataset,
)


def test_train_prod_writes_checkpoint_and_summary(tmp_path: Path) -> None:
    dataset_path = _create_valid_mapped_dataset(tmp_path / "valid_train_prod.mapped")
    checkpoint_path = tmp_path / "translator_train_prod.pt"
    summary_path = tmp_path / "translator_train_prod.json"

    config = build_train_prod_config(
        dataset_path=dataset_path,
        emb_dim=32,
        hidden_dim=64,
        num_heads=4,
        num_layers=2,
        dropout=0.0,
        lr=1e-3,
        batch_size=32,
        epochs=1,
        seed=7,
        max_examples=128,
        shuffle=False,
        log_every=1000,
        device="cpu",
        checkpoint_path=checkpoint_path,
        summary_path=summary_path,
    )
    out = run_train_prod(config)

    assert checkpoint_path.is_file()
    assert summary_path.is_file()
    assert out["checkpoint_path"] == str(checkpoint_path)
    assert out["summary_path"] == str(summary_path)
