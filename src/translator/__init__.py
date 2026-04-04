"""Public package API for the translator project.

The three central objects are:
- Trainer trains the package's Transformer-based model.
- Translator runs inference with the trained model.
- Under the hood is Seq2Seq, a Transformer-based sequence-to-sequence model.
"""

from .api import check_dataset, train
from .inference import Translator
from .training import DataLoaderConfig, ModelConfig, TrainConfig, TrainingSummary

__all__ = [
    "train",
    "check_dataset",
    "Translator",
    "DataLoaderConfig",
    "ModelConfig",
    "TrainConfig",
    "TrainingSummary",
]
