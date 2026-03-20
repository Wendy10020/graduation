import torch
from typing import Optional
from .base_augmentation import BaseAugmentation

class RandomShifter(BaseAugmentation):
    """随机平移增强"""
    
    def __init__(self, shift_backward_max: int, shift_forward_max: int, 
                 sequence_length: int, do_prob: float = 1.0):
        super().__init__(do_prob)
        self.shift_backward_max = abs(shift_backward_max)
        self.shift_forward_max = shift_forward_max
        self.sequence_length = sequence_length
        
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        if x.dim() == 3:
            return torch.stack([self._shift_single(xi) for xi in x]), y
        else:
            return self._shift_single(x), y
    
    def _shift_single(self, x: torch.Tensor) -> torch.Tensor:
        if not self.check_proba():
            return x
            
        start = torch.randint(0, self.shift_backward_max + self.shift_forward_max, (1,)).item()
        
        original_shape = x.shape
        
        if len(original_shape) == 2:
            if original_shape[0] == self.sequence_length:
                x = x.permute(1, 0)
                need_permute = True
            else:
                need_permute = False
        else:
            raise ValueError(f"Unsupported shape: {original_shape}")
        
        x = torch.nn.functional.pad(x, (self.shift_backward_max, self.shift_forward_max))
        x = x[:, start:start + self.sequence_length]
        
        if need_permute:
            x = x.permute(1, 0)
            
        return x