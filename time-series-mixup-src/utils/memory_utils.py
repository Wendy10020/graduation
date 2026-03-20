import psutil
import torch
import gc

def log_memory_usage(stage: str = ""):
    """记录内存使用情况"""
    cpu_mem = psutil.Process().memory_info().rss / 1024**3
    
    if torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024**3
        gpu_cached = torch.cuda.memory_reserved() / 1024**3
        print(f"[{stage}] CPU: {cpu_mem:.2f}GB, GPU: {gpu_mem:.2f}GB, Cache: {gpu_cached:.2f}GB")
    else:
        print(f"[{stage}] CPU: {cpu_mem:.2f}GB")

def clear_memory():
    """清理内存"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()