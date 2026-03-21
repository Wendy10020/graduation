# data/uae_manager.py
#再次修改路径

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
    """UEA数据集管理器 - 适配标准UEA格式"""
    
    def __init__(self, dataset_name: str, data_dir: str = None):
        """
        Args:
            dataset_name: 数据集名称，如 "BasicMotions"
            data_dir: Kaggle Dataset路径，如 "/kaggle/input/datasets/batinspring/uea-multivariate-2018/Multivariate_ts"
        """
        self.dataset_name = dataset_name
        
        # 如果未指定data_dir，尝试常见的Kaggle路径
        if data_dir is None:
            possible_paths = [
                "/kaggle/input/uea-multivariate-2018/Multivariate_ts",
                "/kaggle/input/datasets/batinspring/uea-multivariate-2018/Multivariate_ts",
                "/kaggle/input/uea-multivariate-2018",
                "/kaggle/working/data/Multivariate_ts"
            ]
            for path in possible_paths:
                if Path(path).exists():
                    data_dir = path
                    print(f"Auto-detected dataset path: {data_dir}")
                    break
        
        if data_dir is None:
            raise FileNotFoundError(
                "Cannot find UEA dataset. Please specify data_dir.\n"
                "Example: UAEDatasetManager('BasicMotions', data_dir='/kaggle/input/your-dataset/Multivariate_ts')"
            )
        
        self.data_dir = Path(data_dir)
        print(f"Using dataset path: {self.data_dir}")
        
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self.n_channels = None
        self.n_timesteps = None
        self.n_classes = None
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """加载训练和测试数据"""
        
        # 标准UEA格式: /path/DatasetName/DatasetName_TRAIN.ts
        dataset_folder = self.data_dir / self.dataset_name
        train_path = dataset_folder / f"{self.dataset_name}_TRAIN.ts"
        test_path = dataset_folder / f"{self.dataset_name}_TEST.ts"
        
        if not train_path.exists():
            raise FileNotFoundError(
                f"Dataset {self.dataset_name} not found at {train_path}\n"
                f"Expected structure: {self.data_dir}/{self.dataset_name}/{self.dataset_name}_TRAIN.ts\n"
                f"Please check that your Kaggle Dataset has this structure."
            )
        
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
    
    def get_dataset_info(self) -> dict:
        """获取数据集信息"""
        return {
            'name': self.dataset_name,
            'n_classes': self.n_classes,
            'n_channels': self.n_channels,
            'n_timesteps': self.n_timesteps,
            'train_samples': len(self.train_labels) if self.train_labels is not None else 0,
            'test_samples': len(self.test_labels) if self.test_labels is not None else 0,
            'class_distribution': self._get_class_distribution()
        }
    
    def _get_class_distribution(self) -> dict:
        """获取类别分布"""
        if self.train_labels is None:
            return {}
        unique, counts = np.unique(self.train_labels, return_counts=True)
        return {int(u): int(c) for u, c in zip(unique, counts)}