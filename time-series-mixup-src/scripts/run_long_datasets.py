#!/usr/bin/env python
"""
运行长数据集实验（仅InceptionTime，长度>512）
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
    print("Running Experiments on Long Datasets (length > 512)")
    print("Only InceptionTime will be used")
    print("="*60)
    
    # 加载配置
    config_loader = ConfigLoader('/kaggle/working/configs')
    config = config_loader.load_config('long_datasets_config.yaml')
    
    # 只保留长数据集
    long_datasets = []
    for ds in config['datasets']:
        if ds['n_timesteps'] > 512:
            long_datasets.append(ds)
    
    config['datasets'] = long_datasets
    
    # 只保留InceptionTime
    for model in config['models']:
        if model['name'] != 'InceptionTime':
            model['enabled'] = False
    
    print(f"Total long datasets: {len(long_datasets)}")
    print(f"Datasets: {[ds['name'] for ds in long_datasets]}")
    
    # 运行实验
    runner = ExperimentRunner(config, '/kaggle/working/results/long')
    results = runner.run_all_experiments()
    
    print(f"\nCompleted! Results: {len(results)} experiments")
    print(f"Results saved to /kaggle/working/results/long")

if __name__ == "__main__":
    main()