#!/usr/bin/env python
"""
运行短数据集实验（长度≤512）
"""

import os
import sys
sys.path.insert(0, '/kaggle/working')

import torch
from utils.config_loader import ConfigLoader
from training.experiment_runner import ExperimentRunner
from utils.memory_utils import log_memory_usage, clear_memory

def main():
    print("="*60)
    print("Running Experiments on Short Datasets (length ≤ 512)")
    print("="*60)
    
    # 加载配置
    config_loader = ConfigLoader('/kaggle/working/configs')
    config = config_loader.load_config('short_datasets_config.yaml')
    
    # 只保留短数据集
    short_datasets = []
    for ds in config['datasets']:
        if ds['n_timesteps'] <= 512:
            short_datasets.append(ds)
    
    config['datasets'] = short_datasets
    
    print(f"Total short datasets: {len(short_datasets)}")
    print(f"Datasets: {[ds['name'] for ds in short_datasets]}")
    
    # 运行实验
    runner = ExperimentRunner(config, '/kaggle/working/results/short')
    results = runner.run_all_experiments()
    
    print(f"\nCompleted! Results: {len(results)} experiments")
    print(f"Results saved to /kaggle/working/results/short")

if __name__ == "__main__":
    main()