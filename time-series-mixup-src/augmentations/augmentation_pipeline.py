import torch
from typing import List, Optional, Dict, Any
from .base_augmentation import BaseAugmentation
from .random_shift import RandomShifter
from .window_warp import WindowWarp
from .cutout import Cutout
from .cutmix import Cutmix
from .mixup import Mixup

class AugmentationPipeline:
    """增强管道"""
    
    def __init__(self, augmentations: List[BaseAugmentation]):
        self.augmentations = augmentations
        
    def __call__(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        for aug in self.augmentations:
            if y is not None:
                result = aug(x, y)
                if len(result) == 3:
                    x, y, _ = result
                else:
                    x, y = result
            else:
                x = aug(x)
        return x, y

def create_augmentation_pipeline(config: Dict[str, Any], 
                                 sequence_length: int, 
                                 n_channels: int) -> AugmentationPipeline:
    """创建增强管道"""
    augmentations = []
    
    if config.get('random_shift', {}).get('enabled', False):
        rs_config = config['random_shift']
        augmentations.append(
            RandomShifter(
                shift_backward_max=rs_config.get('backward', 32),
                shift_forward_max=rs_config.get('forward', 32),
                sequence_length=sequence_length,
                do_prob=rs_config.get('do_prob', 0.5)
            )
        )
    
    if config.get('window_warp', {}).get('enabled', False):
        ww_config = config['window_warp']
        augmentations.append(
            WindowWarp(
                min_window_size=ww_config.get('min_size', sequence_length // 8),
                max_window_size=ww_config.get('max_size', sequence_length // 3),
                scale_factor=ww_config.get('scale_factor', 2.0),
                sequence_length=sequence_length,
                do_prob=ww_config.get('do_prob', 0.5)
            )
        )
    
    if config.get('cutout', {}).get('enabled', False):
        co_config = config['cutout']
        augmentations.append(
            Cutout(
                min_cutout_len=co_config.get('min_len', sequence_length // 2),
                max_cutout_len=co_config.get('max_len', sequence_length),
                channel_drop_prob=co_config.get('channel_drop_prob', 0.3),
                sequence_length=sequence_length,
                n_channels=n_channels,
                do_prob=co_config.get('do_prob', 0.5)
            )
        )
    
    if config.get('cutmix', {}).get('enabled', False):
        cm_config = config['cutmix']
        augmentations.append(
            Cutmix(
                min_cutmix_len=cm_config.get('min_len', sequence_length // 2),
                max_cutmix_len=cm_config.get('max_len', sequence_length),
                channel_replace_prob=cm_config.get('channel_replace_prob', 0.3),
                sequence_length=sequence_length,
                n_channels=n_channels,
                do_prob=cm_config.get('do_prob', 0.5)
            )
        )
    
    if config.get('mixup', {}).get('enabled', False):
        mx_config = config['mixup']
        augmentations.append(
            Mixup(
                alpha=mx_config.get('alpha', 1.0),
                do_prob=mx_config.get('do_prob', 0.5)
            )
        )
    
    return AugmentationPipeline(augmentations)