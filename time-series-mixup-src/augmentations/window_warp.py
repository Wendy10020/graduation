import torch
from .base_augmentation import BaseAugmentation
from .common import resize_time_series, cut_time_series

class WindowWarp(BaseAugmentation):
    """窗口扭曲增强"""
    
    def __init__(self, min_window_size: int, max_window_size: int, scale_factor: float,
                 sequence_length: int, method: str = 'bilinear', do_prob: float = 1.0):
        super().__init__(do_prob)
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.scale_factor = scale_factor
        self.sequence_length = sequence_length
        self.method = method
        
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        if x.dim() == 3:
            return torch.stack([self._warp_single(xi) for xi in x]), y
        else:
            return self._warp_single(x), y
    
    def _warp_single(self, x: torch.Tensor) -> torch.Tensor:
        if not self.check_proba():
            return x
            
        start, end = self._get_window()
        window_size = end - start
        
        target_size = max(2, int(window_size * self.scale_factor))
        
        window = x[:, start:end] if x.dim() == 2 else x[start:end]
        warped_window = resize_time_series(window, target_size, mode=self.method)
        
        result = cut_time_series(x, start, end, insert=warped_window)
        
        if result.shape[0] > self.sequence_length:
            result = result[:self.sequence_length]
        elif result.shape[0] < self.sequence_length:
            result = torch.nn.functional.pad(result, (0, self.sequence_length - result.shape[0]))
            
        return result
    
    def _get_window(self) -> tuple:
        max_start = self.sequence_length - self.max_window_size
        start = torch.randint(0, max_start + 1, (1,)).item()
        end = start + torch.randint(self.min_window_size, self.max_window_size + 1, (1,)).item()
        return start, end