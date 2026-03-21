# augmentations/adaptive_mixup.py
import torch
import numpy as np
from typing import Optional, Dict, Any, List
from .base_augmentation import BaseAugmentation

class AdaptiveMixup(BaseAugmentation):
    """自适应Mixup增强 - 修复设备问题"""
    
    def __init__(self, class_statistics: dict, alpha: float = 1.0, do_prob: float = 1.0):
        super().__init__(do_prob)
        self.class_statistics = class_statistics
        self.alpha = alpha
        self.class_centroids = None
        self.class_intra_distances = None
        
    def get_strategy_and_lambda(self, label_i, label_j):
        """获取策略和lambda调整系数"""
        if torch.is_tensor(label_i):
            label_i = label_i.item()
        if torch.is_tensor(label_j):
            label_j = label_j.item()
        
        class_sizes = self.class_statistics.get('class_distribution', {})
        size_i = class_sizes.get(label_i, 0)
        size_j = class_sizes.get(label_j, 0)
        
        avg_size = np.mean(list(class_sizes.values())) if class_sizes else 1
        is_minority = min(size_i, size_j) < 0.3 * avg_size
        
        if label_i == label_j:
            return 'intra_class', 0.9
        elif is_minority:
            return 'intra_class', 0.9
        else:
            return 'inter_class', 0.5
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """执行自适应mixup"""
        if not self.check_proba():
            return x, y, None
            
        batch_size = x.shape[0]
        device = x.device  # 获取输入数据的设备
        
        # 在正确的设备上生成随机排列
        perm = torch.randperm(batch_size, device=device)
        x_shuffled = x[perm]
        y_shuffled = y[perm]
        
        x_mixed = torch.zeros_like(x)
        y_mixed = torch.zeros_like(y)
        mixup_info = []
        
        for i in range(batch_size):
            label_i = y[i]
            label_j = y_shuffled[i]
            
            strategy, lam_adj = self.get_strategy_and_lambda(label_i, label_j)
            
            # 在正确的设备上生成lambda
            lam = np.random.beta(self.alpha, self.alpha)
            lam = lam * lam_adj
            
            if strategy == 'intra_class' and label_i != label_j:
                same_class_indices = torch.where(y == label_i)[0]
                if len(same_class_indices) > 1:
                    other_indices = same_class_indices[same_class_indices != i]
                    if len(other_indices) > 0:
                        j = other_indices[torch.randperm(len(other_indices), device=device)[0]]
                        x_j = x[j]
                        y_j = y[j]
                    else:
                        x_j = x_shuffled[i]
                        y_j = y_shuffled[i]
                else:
                    x_j = x_shuffled[i]
                    y_j = y_shuffled[i]
            else:
                x_j = x_shuffled[i]
                y_j = y_shuffled[i]
            
            x_mixed[i] = lam * x[i] + (1 - lam) * x_j
            y_mixed[i] = lam * y[i] + (1 - lam) * y_j
            
            mixup_info.append({
                'strategy': strategy,
                'lambda': lam,
                'class_pair': (label_i.item() if torch.is_tensor(label_i) else label_i,
                              label_j.item() if torch.is_tensor(label_j) else label_j)
            })
        
        return x_mixed, y_mixed, mixup_info