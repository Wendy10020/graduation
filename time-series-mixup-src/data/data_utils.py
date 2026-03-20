import numpy as np
import torch
from typing import Tuple, Optional

def normalize_data(data: np.ndarray, mean: Optional[np.ndarray] = None, 
                   std: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """标准化数据"""
    if mean is None or std is None:
        mean = np.mean(data, axis=(0, 2), keepdims=True)
        std = np.std(data, axis=(0, 2), keepdims=True)
        std = np.where(std == 0, 1, std)
    
    normalized = (data - mean) / std
    return normalized, mean, std

def pad_sequence(data: np.ndarray, target_length: int, pad_value: float = 0.0) -> np.ndarray:
    """填充序列到目标长度"""
    current_length = data.shape[-1]
    if current_length >= target_length:
        return data[..., :target_length]
    
    pad_width = [(0, 0) for _ in range(data.ndim - 1)] + [(0, target_length - current_length)]
    return np.pad(data, pad_width, constant_values=pad_value)

def split_sequence(data: np.ndarray, segment_length: int, stride: Optional[int] = None) -> np.ndarray:
    """将长序列分割成多个片段"""
    if stride is None:
        stride = segment_length
    
    n_segments = (data.shape[-1] - segment_length) // stride + 1
    segments = []
    
    for i in range(n_segments):
        start = i * stride
        end = start + segment_length
        segments.append(data[..., start:end])
    
    return np.stack(segments, axis=0)

def create_time_mask(sequence_length: int, mask_ratio: float = 0.2) -> torch.Tensor:
    """创建时间掩码"""
    mask_length = int(sequence_length * mask_ratio)
    start = np.random.randint(0, sequence_length - mask_length)
    mask = torch.ones(sequence_length)
    mask[start:start + mask_length] = 0
    return mask

def create_channel_mask(n_channels: int, mask_ratio: float = 0.2) -> torch.Tensor:
    """创建通道掩码"""
    n_mask = int(n_channels * mask_ratio)
    mask_indices = np.random.choice(n_channels, n_mask, replace=False)
    mask = torch.ones(n_channels)
    mask[mask_indices] = 0
    return mask