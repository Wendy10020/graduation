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
    """UEA数据集管理器"""
    
    # 数据集URL
    BASE_URL = "http://www.timeseriesclassification.com/Downloads/Archives/Multivariate2018_ts.zip"
    
    def __init__(self, dataset_name: str, data_dir: str = "/kaggle/working/data"):
        self.dataset_name = dataset_name
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        self.n_channels = None
        self.n_timesteps = None
        self.n_classes = None
        
    def load_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """加载UEA数据集"""
        # 检查是否已下载
        dataset_path = self.data_dir / self.dataset_name
        if not dataset_path.exists():
            self._download_dataset()
        
        # 加载训练集
        train_path = dataset_path / f"{self.dataset_name}_TRAIN.ts"
        test_path = dataset_path / f"{self.dataset_name}_TEST.ts"
        
        if train_path.exists() and test_path.exists():
            x_train, y_train = load_from_tsfile_to_dataframe(str(train_path))
            x_test, y_test = load_from_tsfile_to_dataframe(str(test_path))
        else:
            raise FileNotFoundError(f"Dataset {self.dataset_name} not found")
        
        # 转换为numpy数组
        self.train_data = self._to_numpy_array(x_train)
        self.test_data = self._to_numpy_array(x_test)
        self.train_labels = self._to_numeric_labels(y_train)
        self.test_labels = self._to_numeric_labels(y_test)
        
        # 获取数据集信息
        self.n_channels = self.train_data.shape[1]
        self.n_timesteps = self.train_data.shape[2]
        self.n_classes = len(np.unique(self.train_labels))
        
        return self.train_data, self.train_labels, self.test_data, self.test_labels
    
    def _download_dataset(self):
        """下载并解压UEA数据集"""
        import gdown
        import zipfile
        
        zip_path = self.data_dir / "Multivariate2018_ts.zip"
        
        # 下载数据集
        if not zip_path.exists():
            print(f"Downloading UEA dataset...")
            gdown.download(self.BASE_URL, str(zip_path), quiet=False)
        
        # 解压
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(self.data_dir)
        
        print(f"Dataset extracted to {self.data_dir}")
    
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