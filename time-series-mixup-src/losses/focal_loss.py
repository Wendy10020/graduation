import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """Focal Loss"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        if targets.dim() == 2:
            targets = torch.argmax(targets, dim=1)
        
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weight = (1 - pt) ** self.gamma
        
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple, torch.Tensor)):
                if not isinstance(self.alpha, torch.Tensor):
                    alpha_t = torch.tensor(self.alpha, device=inputs.device)
                else:
                    alpha_t = self.alpha
                alpha_weight = alpha_t[targets]
            else:
                alpha_weight = self.alpha
            focal_weight = alpha_weight * focal_weight
        
        loss = focal_weight * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class AdaptiveFocalLoss(nn.Module):
    """自适应Focal Loss"""
    
    def __init__(self, class_statistics=None, base_gamma=2.0, gamma_range=(1.0, 5.0)):
        super().__init__()
        self.class_statistics = class_statistics
        self.base_gamma = base_gamma
        self.gamma_range = gamma_range
        self.class_gammas = None
        
    def compute_class_gammas(self, class_distribution):
        if class_distribution is None:
            return None
            
        total_samples = sum(class_distribution.values())
        n_classes = len(class_distribution)
        avg_samples = total_samples / n_classes
        
        class_gammas = {}
        for cls, n_samples in class_distribution.items():
            ratio = n_samples / avg_samples
            
            if ratio > 1:
                gamma = self.base_gamma + (ratio - 1) * 0.5
            else:
                gamma = self.base_gamma - (1 - ratio) * 1.0
                
            gamma = max(self.gamma_range[0], min(self.gamma_range[1], gamma))
            class_gammas[cls] = gamma
            
        return class_gammas
    
    def forward(self, inputs, targets):
        if self.class_gammas is None and self.class_statistics is not None:
            self.class_gammas = self.compute_class_gammas(
                self.class_statistics.get('class_distribution', {})
            )
        
        if self.class_gammas is not None:
            losses = []
            for cls in torch.unique(targets):
                mask = targets == cls
                if mask.sum() > 0:
                    cls_inputs = inputs[mask]
                    cls_targets = targets[mask]
                    gamma = self.class_gammas[cls.item()]
                    focal_loss = FocalLoss(gamma=gamma, reduction='none')
                    cls_loss = focal_loss(cls_inputs, cls_targets)
                    losses.append(cls_loss)
            
            if losses:
                loss = torch.cat(losses).mean()
            else:
                loss = torch.tensor(0.0, device=inputs.device)
        else:
            focal_loss = FocalLoss(gamma=self.base_gamma)
            loss = focal_loss(inputs, targets)
            
        return loss