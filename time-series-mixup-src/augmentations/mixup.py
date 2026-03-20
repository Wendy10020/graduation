import torch
import numpy as np
from .base_augmentation import BaseAugmentation

class Mixup(BaseAugmentation):
    """标准Mixup增强"""
    
    def __init__(self, alpha: float = 1.0, do_prob: float = 1.0):
        super().__init__(do_prob)
        self.alpha = alpha
        
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if not self.check_proba():
            return x, y
            
        batch_size = x.shape[0]
        
        perm = torch.randperm(batch_size)
        x_shuffled = x[perm]
        y_shuffled = y[perm]
        
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample((batch_size,))
        lam = lam.view(-1, *([1] * (x.dim() - 1)))
        
        x_mixed = lam * x + (1 - lam) * x_shuffled
        y_mixed = lam.squeeze() * y + (1 - lam.squeeze()) * y_shuffled
        
        return x_mixed, y_mixed