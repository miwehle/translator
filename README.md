# Minimaler Transformer-Translator (PyTorch)

Als Beschreibung der zentralen öffentlichen Objekte siehe
`src/translator/__init__.py`.

Dieses Projekt enthaelt einen bewusst einfachen Transformer-Translator mit:

- Transformer-Encoder mit Multi-Head Self-Attention
- Transformer-Decoder mit Masked Self-Attention und Cross-Attention
- Pre-Norm (LayerNorm vor jedem Sub-Block)
- Feed-Forward-Netzwerken pro Block

## Struktur

- `simple_attention_translator.py` (Entry-Point)
- `src/translator/constants.py` (Sondertokens)
- `src/translator/data.py` (Vokabular, Dataset, Collate, Toy-Korpus)
- `src/translator/model.py` (Transformer-Modell)
- `src/translator/train.py` (Training, CLI, Beispiel-Inferenz)

## Start

```bash
python simple_attention_translator.py --epochs 120
```

## Checkpoint und Inferenz

Training speichert automatisch ein Checkpoint unter `checkpoints/translator.pt` (anpassbar mit `--checkpoint-path`).

### Training runs

For regular training, use `train_config.experiment_id` to group related runs:

`training_runs/de-en-translator/r1`, `training_runs/de-en-translator/r2`, ...

This keeps the top-level `training_runs` directory small and makes related runs easier to find. For simple ad-hoc runs, `experiment_id` can be omitted; those runs are stored directly as `training_runs/r1`, `training_runs/r2`, ...

Einzelsatz uebersetzen:

```bash
python simple_attention_translator.py --checkpoint-path checkpoints/translator.pt --translate "ich bin muede"
```

Interaktiv testen:

```bash
python simple_attention_translator.py --checkpoint-path checkpoints/translator.pt --interactive
```

## Wichtige Parameter

- `--epochs` (default: `200`)
- `--batch-size` (default: `4`)
- `--emb-dim` (default: `64`)
- `--hidden-dim` (default: `64`, Feed-Forward-Dimension)
- `--lr` (default: `1e-3`)
- `--num-heads` (default: `4`)
- `--num-layers` (default: `2`)
- `--dropout` (default: `0.1`)
- `--checkpoint-path` (default: `checkpoints/translator.pt`)
- `--translate` (default: `None`)
- `--interactive` (Flag)
- `--max-len` (default: `30`, nur Inferenz)

## Hinweis

Der Datensatz ist absichtlich sehr klein (Toy-Parallelkorpus in der Datei), damit das Beispiel schnell verstaendlich und trainierbar bleibt.
