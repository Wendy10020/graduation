import torch
from .base_augmentation import BaseAugmentation

class Cutmix(BaseAugmentation):
    """Cutmix增强"""
    
    def __init__(self, min_cutmix_len: int, max_cutmix_len: int, channel_replace_prob: float,
                 sequence_length: int, n_channels: int, do_prob: float = 1.0):
        super().__init__(do_prob)
        self.min_cutmix_len = min_cutmix_len
        self.max_cutmix_len = max_cutmix_len
        self.channel_replace_prob = channel_replace_prob
        self.sequence_length = sequence_length
        self.n_channels = n_channels
        
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        if not self.check_proba():
            return x, y
            
        batch_size = x.shape[0]
        perm = torch.randperm(batch_size)
        x_shuffled = x[perm]
        y_shuffled = y[perm] if y is not None else None
        
        masks = torch.stack([self._get_cutmix_mask(x.device) for _ in range(batch_size)])
        
        x_mixed = torch.where(masks == 1, x, x_shuffled)
        
        if y is not None:
            mix_ratios = masks.mean(dim=(1, 2))
            y_mixed = (1 - mix_ratios).unsqueeze(1) * y + mix_ratios.unsqueeze(1) * y_shuffled
            return x_mixed, y_mixed
        else:
            return x_mixed, None
    
    def _get_cutmix_mask(self, device) -> torch.Tensor:
        start = torch.randint(0, self.sequence_length - self.max_cutmix_len, (1,)).item()
        end = start + torch.randint(self.min_cutmix_len, self.max_cutmix_len + 1, (1,)).item()
        
        time_mask = torch.ones(self.sequence_length, device=device)
        time_mask[start:end] = 0
        
        channel_mask = (torch.rand(self.n_channels, device=device) > self.channel_replace_prob).float()
        
        mask = channel_mask.unsqueeze(1) * time_mask.unsqueeze(0)
        
        return mask