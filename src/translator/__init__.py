"""Public package API for the translator project.

The three central objects are:
- Trainer trains the package's Transformer-based model.
- Translator runs inference with the trained model.
- Under the hood is Seq2Seq, a Transformer-based sequence-to-sequence model.

This package is also meant as a small lab for understanding and applying
the following ideas in an implementation-oriented way, in the spirit of the
MIT motto "Mens et Manus":
the 2014 paper by Sutskever, Vinyals and Le,
"Sequence to Sequence Learning with Neural Networks"
(https://arxiv.org/abs/1409.3215),
Bahdanau, Cho and Bengio 2014
(https://arxiv.org/abs/1409.0473),
and the Transformer paper by Vaswani et al. 2017
(https://arxiv.org/abs/1706.03762),
including sequence transduction models in encoder-decoder configuration.
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
