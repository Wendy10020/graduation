"""
Time Series Mixup Benchmark
A comprehensive benchmark for mixup strategies on multivariate time series classification.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from .data import UAEDatasetManager, BaseDatasetManager
from .models import ModelFactory, InceptionTime, SimpleRNN, SimpleMHSA
from .augmentations import Mixup, AdaptiveMixup
from .losses import FocalLoss
from .training import Trainer, ExperimentRunner
from .utils import ConfigLoader, log_memory_usage, clear_memory

__all__ = [
    'UAEDatasetManager',
    'BaseDatasetManager',
    'ModelFactory',
    'InceptionTime',
    'SimpleRNN',
    'SimpleMHSA',
    'Mixup',
    'AdaptiveMixup',
    'FocalLoss',
    'Trainer',
    'ExperimentRunner',
    'ConfigLoader',
    'log_memory_usage',
    'clear_memory'
]