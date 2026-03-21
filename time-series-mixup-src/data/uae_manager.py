# data/uae_manager.py - 修改后的版本

import numpy as np
import pandas as pd
from sktime.datasets import load_from_tsfile_to_dataframe
from sktime.datatypes._panel._convert import from_nested_to_3d_numpy
from pathlib import Path
from typing import Tuple
import warnings
warnings.filterwarnings('ignore')

from .dataset_manager import BaseDatasetManager

class UAEDatasetManager(BaseDatasetManager):
    """UEA数据集管理器 - 使用Kaggle Dataset"""
    
    def __init__(self, dataset_name: str, data_dir: str = "/kaggle/input/uea-multivariate-2018/Multivariate_ts"):
        """
        Args:
            dataset_name: 数据集名称
            data_dir: Kaggle Dataset路径
        """
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self.n_channels = None
        self.n_timesteps = None
        self.n_classes = None
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """加载训练和测试数据"""
        # 构建文件路径
        dataset_path = self.data_dir / self.dataset_name
        
        # 尝试不同的路径格式
        train_path = dataset_path / f"{self.dataset_name}_TRAIN.ts"
        test_path = dataset_path / f"{self.dataset_name}_TEST.ts"
        
        if not train_path.exists():
            # 尝试另一种路径格式
            train_path = self.data_dir / f"{self.dataset_name}_TRAIN.ts"
            test_path = self.data_dir / f"{self.dataset_name}_TEST.ts"
        
        if not train_path.exists():
            raise FileNotFoundError(f"Dataset {self.dataset_name} not found at {train_path}")
        
        print(f"Loading {self.dataset_name} from {train_path}")
        
        # 使用sktime加载数据
        x_train, y_train = load_from_tsfile_to_dataframe(str(train_path))
        x_test, y_test = load_from_tsfile_to_dataframe(str(test_path))
        
        # 转换为numpy数组
        self.train_data = self._to_numpy_array(x_train)
        self.test_data = self._to_numpy_array(x_test)
        self.train_labels = self._to_numeric_labels(y_train)
        self.test_labels = self._to_numeric_labels(y_test)
        
        # 获取数据集信息
        self.n_channels = self.train_data.shape[1]
        self.n_timesteps = self.train_data.shape[2]
        self.n_classes = len(np.unique(self.train_labels))
        
        print(f"Loaded: {self.train_data.shape[0]} train, {self.test_data.shape[0]} test")
        print(f"Channels: {self.n_channels}, Timesteps: {self.n_timesteps}, Classes: {self.n_classes}")
        
        return self.train_data, self.train_labels, self.test_data, self.test_labels
    
    def _to_numpy_array(self, sktime_data: pd.DataFrame) -> np.ndarray:
        """将sktime格式转换为 (n_samples, n_channels, n_timesteps)"""
        arr = from_nested_to_3d_numpy(sktime_data)
        # sktime输出为 (n_samples, n_timesteps, n_channels)
        arr = np.transpose(arr, (0, 2, 1))
        return arr.astype(np.float32)
    
    def _to_numeric_labels(self, labels: pd.Series) -> np.ndarray:
        """将标签转换为数值"""
        unique = np.unique(labels)
        mapping = {label: i for i, label in enumerate(unique)}
        return np.array([mapping[l] for l in labels])