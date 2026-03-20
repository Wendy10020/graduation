"""
Data augmentation module for time series classification.
"""

from .base_augmentation import BaseAugmentation
from .random_shift import RandomShifter
from .window_warp import WindowWarp
from .cutout import Cutout
from .cutmix import Cutmix
from .mixup import Mixup
from .adaptive_mixup import AdaptiveMixup
from .augmentation_pipeline import AugmentationPipeline, create_augmentation_pipeline

__all__ = [
    'BaseAugmentation',
    'RandomShifter',
    'WindowWarp',
    'Cutout',
    'Cutmix',
    'Mixup',
    'AdaptiveMixup',
    'AugmentationPipeline',
    'create_augmentation_pipeline'
]