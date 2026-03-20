"""
Common utility functions for augmentations.
"""

import torch
import torch.nn.functional as F

def resize_time_series(series: torch.Tensor, new_length: int, mode: str = 'bilinear') -> torch.Tensor:
    """
    Resize a time series using interpolation.
    
    Args:
        series: Input tensor of shape [channels, length] or [length]
        new_length: Target length
        mode: Interpolation mode ('bilinear' or 'nearest')
    
    Returns:
        Resized tensor
    """
    original_shape = series.shape
    
    if len(original_shape) == 1:
        series = series.unsqueeze(0).unsqueeze(0)
        need_reshape = True
    elif len(original_shape) == 2:
        # Assume shape [channels, length]
        series = series.unsqueeze(0)
        need_reshape = True
    else:
        need_reshape = False
    
    if mode == 'bilinear':
        series = F.interpolate(series, size=new_length, mode='linear', align_corners=False)
    elif mode == 'nearest':
        series = F.interpolate(series, size=new_length, mode='nearest')
    else:
        raise ValueError(f"Unsupported mode: {mode}")
    
    if need_reshape:
        series = series.squeeze(0)
        if len(original_shape) == 1:
            series = series.squeeze(0)
    
    return series

def pad_to_length(series: torch.Tensor, target_length: int, padding_value: float = 0.0) -> torch.Tensor:
    """
    Pad a time series to target length.
    
    Args:
        series: Input tensor
        target_length: Desired length
        padding_value: Value to pad with
    
    Returns:
        Padded tensor
    """
    current_length = series.shape[-1]
    if current_length >= target_length:
        return series[..., :target_length]
    
    pad_size = target_length - current_length
    padding = torch.full((*series.shape[:-1], pad_size), padding_value, device=series.device)
    return torch.cat([series, padding], dim=-1)

def cut_time_series(series: torch.Tensor, cut_start: int, cut_end: int, 
                    insert: torch.Tensor = None) -> torch.Tensor:
    """
    Cut a segment from a time series and optionally insert another segment.
    
    Args:
        series: Input tensor [..., length]
        cut_start: Start index to cut
        cut_end: End index to cut
        insert: Optional tensor to insert
    
    Returns:
        Modified tensor
    """
    original_length = series.shape[-1]
    
    parts = [series[..., :cut_start]]
    if insert is not None:
        parts.append(insert)
    parts.append(series[..., cut_end:])
    
    result = torch.cat(parts, dim=-1)
    
    if result.shape[-1] < original_length:
        result = pad_to_length(result, original_length)
    elif result.shape[-1] > original_length:
        result = result[..., :original_length]
    
    return result

def check_proba(proba: float) -> bool:
    """
    Check if a random probability is less than given probability.
    
    Args:
        proba: Probability threshold
    
    Returns:
        True if random number < proba
    """
    return torch.rand(1).item() <= proba