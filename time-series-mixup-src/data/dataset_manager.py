from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional, Dict, Any

class TimeSeriesDataset(Dataset):
    """时间序列数据集类"""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

class BaseDatasetManager(ABC):
    """数据集管理器抽象基类"""
    
    @abstractmethod
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """加载训练和测试数据"""
        pass
    
    def get_dataloaders(self, batch_size: int, shuffle: bool = True,
                        num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
        """获取训练和测试数据加载器"""
        if self.train_data is None or self.test_data is None:
            self.load_data()
            
        train_dataset = TimeSeriesDataset(self.train_data, self.train_labels)
        test_dataset = TimeSeriesDataset(self.test_data, self.test_labels)
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, pin_memory=True, drop_last=True
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True
        )
        
        return train_loader, test_loader
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """获取数据集信息"""
        return {
            "name": self.dataset_name,
            "n_classes": self.n_classes,
            "n_channels": self.n_channels,
            "n_timesteps": self.n_timesteps,
            "train_samples": len(self.train_labels) if self.train_labels is not None else 0,
            "test_samples": len(self.test_labels) if self.test_labels is not None else 0,
            "class_distribution": self._get_class_distribution()
        }
    
    def _get_class_distribution(self) -> Dict[int, int]:
        """获取类别分布"""
        if self.train_labels is None:
            return {}
        unique, counts = np.unique(self.train_labels, return_counts=True)
        return dict(zip(unique, counts))