"""Public package API for the translator project.

The three central objects are:
- Trainer trains the package's Transformer-based model.
- Translator runs inference with the trained model.
- Under the hood is Seq2Seq, a Transformer-based sequence-to-sequence model.

This package is also meant as a small lab for understanding and applying
the following ideas in an implementation-oriented way, in the spirit of the
MIT motto "Mens et Manus":
- the 2014 paper by Sutskever, Vinyals and Le,
  "Sequence to Sequence Learning with Neural Networks"
  (https://arxiv.org/abs/1409.3215),
- Bahdanau, Cho and Bengio, 2014
  "Neural Machine Translation by Jointly Learning to Align and Translate"
  (https://arxiv.org/abs/1409.0473),
- the Transformer paper "Attention Is All You Need" by Vaswani et al. 2017
  on transformer models in encoder-decoder configuration
  (https://arxiv.org/abs/1706.03762).
"""

from .api import comet_score, preflight_check, train
from .evaluation.config import CometScoreRunConfig, DatasetConfig, MappingConfig
from .inference import Translator
from .training import DataLoaderConfig, ModelConfig, PreflightCheckRunConfig, TrainRunConfig
from .training.trainer import TrainingSummary

__all__ = [
    "train",
    "comet_score",
    "preflight_check",
    "Translator",
    "CometScoreRunConfig",
    "DatasetConfig",
    "MappingConfig",
    "DataLoaderConfig",
    "ModelConfig",
    "PreflightCheckRunConfig",
    "TrainRunConfig",
    "TrainingSummary",
]
