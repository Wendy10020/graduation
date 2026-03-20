import torch
import numpy as np
from typing import Optional, Dict, Any
from .base_augmentation import BaseAugmentation

class AdaptiveMixup(BaseAugmentation):
    """自适应Mixup增强"""
    
    def __init__(self, class_statistics: dict, alpha: float = 1.0, do_prob: float = 1.0):
        super().__init__(do_prob)
        self.class_statistics = class_statistics
        self.alpha = alpha
        self.class_centroids = None
        self.class_intra_distances = None
        
    def compute_class_features(self, data_loader, model=None):
        """计算类别特征"""
        if model is None:
            return
            
        all_features = []
        all_labels = []
        
        model.eval()
        with torch.no_grad():
            for x, y in data_loader:
                if hasattr(model, 'extract_features'):
                    features = model.extract_features(x)
                else:
                    features = x.view(x.size(0), -1)
                all_features.append(features)
                all_labels.append(y)
        
        all_features = torch.cat(all_features)
        all_labels = torch.cat(all_labels)
        
        self.class_centroids = {}
        self.class_intra_distances = {}
        
        for c in range(self.class_statistics['n_classes']):
            mask = all_labels == c
            if mask.sum() > 0:
                class_features = all_features[mask]
                centroid = class_features.mean(dim=0)
                self.class_centroids[c] = centroid
                
                distances = torch.norm(class_features - centroid, dim=1)
                self.class_intra_distances[c] = distances.mean().item()
    
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
        
        if self.class_centroids is not None and label_i in self.class_centroids and label_j in self.class_centroids:
            centroid_i = self.class_centroids[label_i]
            centroid_j = self.class_centroids[label_j]
            inter_distance = torch.norm(centroid_i - centroid_j).item()
            similarity = 1.0 / (1.0 + inter_distance)
        else:
            similarity = 0.5
        
        if is_minority or label_i == label_j:
            return 'intra_class', 0.9
        elif similarity > 0.7:
            return 'inter_class_similar', 0.7
        else:
            return 'inter_class_distinct', 0.3
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        """执行自适应mixup"""
        if not self.check_proba():
            return x, y, None
            
        batch_size = x.shape[0]
        perm = torch.randperm(batch_size)
        x_shuffled = x[perm]
        y_shuffled = y[perm]
        
        x_mixed = torch.zeros_like(x)
        y_mixed = torch.zeros_like(y)
        mixup_info = []
        
        for i in range(batch_size):
            label_i = y[i]
            label_j = y_shuffled[i]
            
            strategy, lam_adj = self.get_strategy_and_lambda(label_i, label_j)
            
            lam = np.random.beta(self.alpha, self.alpha)
            lam = lam * lam_adj
            
            if strategy == 'intra_class' and label_i != label_j:
                same_class_indices = torch.where(y == label_i)[0]
                if len(same_class_indices) > 1:
                    other_indices = same_class_indices[same_class_indices != i]
                    if len(other_indices) > 0:
                        j = other_indices[torch.randperm(len(other_indices))[0]]
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
                'class_pair': (label_i, label_j)
            })
        
        return x_mixed, y_mixed, mixup_info