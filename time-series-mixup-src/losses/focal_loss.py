# losses/focal_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Union

class FocalLoss(nn.Module):
    """
    Focal Loss for multi-class classification
    
    Reference: "Focal Loss for Dense Object Detection"
    
    Args:
        alpha: Class weights, can be scalar or list/tensor
        gamma: Focusing parameter
        reduction: Loss reduction method ('mean', 'sum', 'none')
    """
    
    def __init__(self, alpha: Optional[Union[float, list, torch.Tensor]] = None, 
                 gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, 
                mixup_info: Optional[Dict] = None) -> torch.Tensor:
        """
        Args:
            inputs: Model outputs [batch_size, num_classes]
            targets: Ground truth labels [batch_size] or one-hot [batch_size, num_classes]
            mixup_info: Optional mixup information (for compatibility)
        
        Returns:
            Loss value
        """
        # Convert one-hot to indices if needed
        if targets.dim() == 2:
            targets = torch.argmax(targets, dim=1)
        
        # Compute cross entropy loss
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        
        # Compute p_t
        pt = torch.exp(-ce_loss)
        
        # Compute focal weight
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha_t = torch.tensor(self.alpha, device=inputs.device)
                alpha_weight = alpha_t[targets]
            elif isinstance(self.alpha, torch.Tensor):
                alpha_weight = self.alpha[targets]
            else:
                alpha_weight = self.alpha
            focal_weight = alpha_weight * focal_weight
        
        # Compute final loss
        loss = focal_weight * ce_loss
        
        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AdaptiveFocalLoss(nn.Module):
    """
    Adaptive Focal Loss that adjusts gamma based on class statistics
    
    Args:
        class_statistics: Dictionary containing class distribution and other info
        base_gamma: Base gamma value
        gamma_range: Range for gamma values (min, max)
    """
    
    def __init__(self, class_statistics: Optional[Dict] = None, 
                 base_gamma: float = 2.0, gamma_range: tuple = (1.0, 5.0)):
        super().__init__()
        self.class_statistics = class_statistics
        self.base_gamma = base_gamma
        self.gamma_range = gamma_range
        self.class_gammas = None
        
        # Pre-compute class-specific gammas if class statistics available
        if class_statistics is not None:
            self._compute_class_gammas()
    
    def _compute_class_gammas(self):
        """Compute gamma values for each class based on class distribution"""
        class_distribution = self.class_statistics.get('class_distribution', {})
        if not class_distribution:
            self.class_gammas = None
            return
        
        total_samples = sum(class_distribution.values())
        n_classes = len(class_distribution)
        avg_samples = total_samples / n_classes if n_classes > 0 else 1
        
        class_gammas = {}
        for cls, n_samples in class_distribution.items():
            # Calculate ratio relative to average
            ratio = n_samples / avg_samples if avg_samples > 0 else 1
            
            # Adjust gamma: majority classes get higher gamma, minority get lower
            if ratio > 1:
                # Majority class: increase gamma to focus more on hard examples
                gamma = self.base_gamma + (ratio - 1) * 0.5
            else:
                # Minority class: decrease gamma to prevent over-focusing
                gamma = self.base_gamma - (1 - ratio) * 1.0
            
            # Clamp to range
            gamma = max(self.gamma_range[0], min(self.gamma_range[1], gamma))
            class_gammas[cls] = gamma
        
        self.class_gammas = class_gammas
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, 
                mixup_info: Optional[Dict] = None) -> torch.Tensor:
        """
        Forward pass with adaptive gamma per class
        
        Args:
            inputs: Model outputs [batch_size, num_classes]
            targets: Ground truth labels [batch_size]
            mixup_info: Optional mixup information
        
        Returns:
            Loss value
        """
        # Convert one-hot to indices if needed
        if targets.dim() == 2:
            targets = torch.argmax(targets, dim=1)
        
        # If no class-specific gammas, use standard focal loss
        if self.class_gammas is None:
            focal_loss = FocalLoss(gamma=self.base_gamma)
            return focal_loss(inputs, targets)
        
        # Compute loss per class with different gamma values
        losses = []
        unique_classes = torch.unique(targets)
        
        for cls in unique_classes:
            mask = targets == cls
            if mask.sum() > 0:
                cls_inputs = inputs[mask]
                cls_targets = targets[mask]
                gamma = self.class_gammas[cls.item()]
                
                # Compute focal loss for this class
                ce_loss = F.cross_entropy(cls_inputs, cls_targets, reduction='none')
                pt = torch.exp(-ce_loss)
                focal_weight = (1 - pt) ** gamma
                cls_loss = focal_weight * ce_loss
                losses.append(cls_loss)
        
        if losses:
            loss = torch.cat(losses).mean()
        else:
            loss = torch.tensor(0.0, device=inputs.device)
        
        return loss


class MixupAwareFocalLoss(nn.Module):
    """
    Focal loss that adapts based on mixup strategy
    
    Args:
        base_gamma: Base gamma value
        mixup_adjustment: Whether to adjust loss based on mixup lambda
    """
    
    def __init__(self, base_gamma: float = 2.0, mixup_adjustment: bool = True):
        super().__init__()
        self.base_gamma = base_gamma
        self.mixup_adjustment = mixup_adjustment
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor, 
                mixup_info: Optional[Dict] = None) -> torch.Tensor:
        """
        Forward pass with mixup-aware adjustment
        
        Args:
            inputs: Model outputs [batch_size, num_classes]
            targets: Ground truth labels or mixed labels
            mixup_info: Dictionary with mixup strategy and lambda
        """
        # Determine gamma based on mixup strategy
        if mixup_info is not None and self.mixup_adjustment:
            strategy = mixup_info.get('strategy', '')
            mixup_lambda = mixup_info.get('lambda', 0.5)
            
            if strategy == 'intra_class':
                # Intra-class mixup: use lower gamma
                gamma = self.base_gamma * 0.7
            elif strategy == 'inter_class':
                # Inter-class mixup: use higher gamma to focus on boundaries
                gamma = self.base_gamma * 1.3
            else:
                gamma = self.base_gamma
            
            # Adjust based on lambda: smaller lambda (more mixing) should have higher gamma
            gamma = gamma * (1.0 + (1.0 - mixup_lambda) * 0.5)
            gamma = max(1.0, min(5.0, gamma))
        else:
            gamma = self.base_gamma
        
        # Compute focal loss
        if targets.dim() == 2:
            # Mixed labels (e.g., from mixup)
            # For mixed labels, we need to compute loss for each class separately
            loss = 0.0
            for c in range(targets.size(1)):
                class_target = targets[:, c]
                if class_target.sum() > 0:
                    # Compute focal loss for this class weight
                    ce_loss = F.binary_cross_entropy_with_logits(inputs[:, c], class_target, reduction='none')
                    pt = torch.exp(-ce_loss)
                    focal_weight = (1 - pt) ** gamma
                    loss = loss + (focal_weight * ce_loss).sum()
            return loss / inputs.size(0)
        else:
            # Standard labels
            ce_loss = F.cross_entropy(inputs, targets, reduction='none')
            pt = torch.exp(-ce_loss)
            focal_weight = (1 - pt) ** gamma
            loss = (focal_weight * ce_loss).mean()
            return loss