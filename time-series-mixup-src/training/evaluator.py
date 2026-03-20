"""
模型评估器
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from typing import Dict, Any, List, Tuple


class Evaluator:
    """评估器"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def evaluate(self, test_loader) -> Dict[str, Any]:
        """评估模型"""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                _, predicted = output.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        results = {
            'accuracy': accuracy_score(all_targets, all_preds),
            'f1_macro': f1_score(all_targets, all_preds, average='macro', zero_division=0),
            'f1_weighted': f1_score(all_targets, all_preds, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(all_targets, all_preds),
            'predictions': all_preds,
            'targets': all_targets
        }
        
        return results
    
    def evaluate_by_class(self, test_loader) -> Dict[int, float]:
        """按类别评估"""
        results = self.evaluate(test_loader)
        cm = results['confusion_matrix']
        
        class_accuracies = {}
        for i in range(cm.shape[0]):
            class_accuracies[i] = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0.0
            
        return class_accuracies