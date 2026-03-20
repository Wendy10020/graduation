import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
from typing import Optional, Dict, Any, Tuple
import time

class Trainer:
    """模型训练器"""
    
    def __init__(self, model: nn.Module, config: Dict[str, Any],
                 class_statistics: Optional[Dict] = None):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 损失函数
        self.criterion = self._create_loss_function(class_statistics)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['hyperparameters']['learning_rate'],
            weight_decay=config['hyperparameters']['weight_decay']
        )
        
        # 学习率调度器
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2
        )
        
        # 混合精度训练
        self.use_amp = config['environment'].get('mixed_precision', False)
        self.scaler = GradScaler() if self.use_amp else None
        
        self.model.to(self.device)
        
    def _create_loss_function(self, class_statistics: Optional[Dict] = None) -> nn.Module:
        """创建损失函数"""
        loss_config = self.config['loss']
        
        if loss_config['type'] == 'focal':
            from losses.focal_loss import FocalLoss
            return FocalLoss(
                gamma=loss_config.get('focal_gamma', 2.0),
                alpha=loss_config.get('focal_alpha', None)
            )
        elif loss_config['type'] == 'adaptive_focal':
            from losses.focal_loss import AdaptiveFocalLoss
            return AdaptiveFocalLoss(class_statistics)
        else:
            return nn.CrossEntropyLoss()
    
    def train_epoch(self, train_loader, augmentation=None) -> Tuple[float, float]:
        """训练一个epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            # 应用数据增强
            mixup_info = None
            if augmentation is not None:
                if hasattr(augmentation, 'forward'):
                    result = augmentation(data, target)
                    if len(result) == 3:
                        data, target, mixup_info = result
                    else:
                        data, target = result
            
            # 前向传播
            self.optimizer.zero_grad()
            
            if self.use_amp:
                with autocast():
                    output = self.model(data)
                    if mixup_info is not None:
                        loss = self.criterion(output, target, mixup_info=mixup_info)
                    else:
                        loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                               self.config['hyperparameters']['gradient_clip'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                if mixup_info is not None:
                    loss = self.criterion(output, target, mixup_info=mixup_info)
                else:
                    loss = self.criterion(output, target)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                              self.config['hyperparameters']['gradient_clip'])
                self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            if mixup_info is None:
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            # 更新进度条
            avg_loss = total_loss / (batch_idx + 1)
            acc = 100. * correct / total if total > 0 else 0
            pbar.set_postfix({'Loss': f'{avg_loss:.4f}', 'Acc': f'{acc:.2f}%'})
        
        return total_loss / len(train_loader), acc
    
    def validate(self, val_loader) -> Tuple[float, float]:
        """验证"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                
                loss = nn.CrossEntropyLoss()(output, target)
                total_loss += loss.item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(val_loader)
        
        return avg_loss, accuracy, all_preds, all_targets
    
    def train(self, train_loader, val_loader, augmentation=None,
              epochs: int = None) -> Dict[str, Any]:
        """完整训练流程"""
        if epochs is None:
            epochs = self.config['hyperparameters']['training_epochs']
        
        best_accuracy = 0
        best_model_state = None
        patience_counter = 0
        early_stopping = self.config['hyperparameters'].get('early_stopping_patience', 10)
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch(train_loader, augmentation)
            
            # 验证
            val_loss, val_acc, _, _ = self.validate(val_loader)
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录历史
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # 保存最佳模型
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"New best model! Accuracy: {val_acc:.2f}%")
            else:
                patience_counter += 1
                if patience_counter >= early_stopping:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        # 加载最佳模型
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        return {
            'best_accuracy': best_accuracy,
            'history': history,
            'model_state': best_model_state
        }