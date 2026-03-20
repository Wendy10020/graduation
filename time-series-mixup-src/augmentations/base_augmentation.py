import torch
from abc import ABC, abstractmethod
from typing import Optional, Tuple

class BaseAugmentation(ABC):
    """增强基类"""
    
    def __init__(self, do_prob: float = 1.0):
        self.do_prob = do_prob
        
    def __call__(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        if self.do_prob < 1.0:
            if torch.rand(1).item() > self.do_prob:
                return x, y
        return self.forward(x, y)
    
    @abstractmethod
    def forward(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        pass
    
    def check_proba(self) -> bool:
        """检查是否执行增强"""
        return torch.rand(1).item() <= self.do_prob