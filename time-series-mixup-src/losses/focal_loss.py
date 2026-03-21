# losses/focal_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any

class FocalLoss(nn.Module):
    """Focal Loss - 支持mixup_info参数"""
    
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets, mixup_info: Optional[Dict] = None):
        """
        Args:
            inputs: 模型输出
            targets: 真实标签
            mixup_info: mixup信息（可选，用于兼容adaptive_mixup）
        """
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
        
        # 如果提供了mixup_info，可以根据策略调整损失
        if mixup_info is not None:
            strategy = mixup_info.get('strategy', '')
            mixup_lambda = mixup_info.get('lambda', 1.0)
            
            if strategy == 'intra_class':
                # 同类mixup：保持原损失
                pass
            elif strategy == 'inter_class':
                # 异类mixup：稍微降低权重
                loss = loss * 0.9
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss