import torch
from typing import Optional
from .base_augmentation import BaseAugmentation

class Cutout(BaseAugmentation):
    """Cutout增强"""
    
    def __init__(self, min_cutout_len: int, max_cutout_len: int, channel_drop_prob: float,
                 sequence_length: int, n_channels: int, do_prob: float = 1.0):
        super().__init__(do_prob)
        self.min_cutout_len = min_cutout_len
        self.max_cutout_len = max_cutout_len
        self.channel_drop_prob = channel_drop_prob
        self.sequence_length = sequence_length
        self.n_channels = n_channels
        
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        if x.dim() == 3:
            return torch.stack([self._cutout_single(xi) for xi in x]), y
        else:
            return self._cutout_single(x), y
    
    def _cutout_single(self, x: torch.Tensor) -> torch.Tensor:
        if not self.check_proba():
            return x
            
        mask = self._get_cutout_mask(x.device)
        
        if x.dim() == 2:
            if x.shape[0] == self.sequence_length:
                return x * mask.permute(1, 0)
            else:
                return x * mask
        else:
            return x * mask
    
    def _get_cutout_mask(self, device) -> torch.Tensor:
        start = torch.randint(0, self.sequence_length - self.max_cutout_len, (1,)).item()
        end = start + torch.randint(self.min_cutout_len, self.max_cutout_len + 1, (1,)).item()
        
        time_mask = torch.ones(self.sequence_length, device=device)
        time_mask[start:end] = 0
        
        channel_mask = (torch.rand(self.n_channels, device=device) > self.channel_drop_prob).float()
        
        mask = channel_mask.unsqueeze(1) * time_mask.unsqueeze(0)
        
        return mask