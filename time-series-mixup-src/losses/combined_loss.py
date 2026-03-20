import torch
import torch.nn as nn
from .focal_loss import FocalLoss, AdaptiveFocalLoss

class MixupAwareLoss(nn.Module):
    """Mixup感知损失"""
    
    def __init__(self, use_focal=True, alpha=0.5, gamma=2.0):
        super().__init__()
        self.use_focal = use_focal
        if use_focal:
            self.base_loss = FocalLoss(gamma=gamma)
        else:
            self.base_loss = nn.CrossEntropyLoss()
        self.alpha = alpha
        
    def forward(self, inputs, targets, is_mixup=False, mixup_lambda=None):
        if self.use_focal and is_mixup:
            focal_loss = FocalLoss(gamma=self.base_loss.gamma * 0.5)
            loss = focal_loss(inputs, targets)
            if mixup_lambda is not None:
                weight = 1.0 / (mixup_lambda + 1e-8)
                loss = loss * torch.clamp(weight, max=10.0)
        else:
            loss = self.base_loss(inputs, targets)
        return loss

class AdaptiveMixupLoss(nn.Module):
    """自适应Mixup损失"""
    
    def __init__(self, class_statistics, base_gamma=2.0):
        super().__init__()
        self.class_statistics = class_statistics
        self.base_gamma = base_gamma
        self.focal_loss = AdaptiveFocalLoss(class_statistics, base_gamma)
        
    def forward(self, inputs, targets, mixup_info=None):
        if mixup_info is None:
            return self.focal_loss(inputs, targets)
        
        strategy = mixup_info['strategy']
        mixup_lambda = mixup_info['lambda']
        
        if strategy == 'intra_class':
            gamma = self.base_gamma * 0.5
            loss_fn = FocalLoss(gamma=gamma)
            loss = loss_fn(inputs, targets)
        elif strategy == 'inter_class_similar':
            gamma = self.base_gamma * 1.5
            loss_fn = FocalLoss(gamma=gamma)
            loss = loss_fn(inputs, targets)
        else:
            loss = self.focal_loss(inputs, targets)
        
        loss = loss * (1.0 - mixup_lambda * 0.5)
        
        return loss