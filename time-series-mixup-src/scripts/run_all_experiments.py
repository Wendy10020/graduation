#!/usr/bin/env python
"""
主实验运行脚本
在Kaggle上运行时，将本文件复制到notebook中执行
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, '/kaggle/working')

from utils.config_loader import ConfigLoader
from training.experiment_runner import ExperimentRunner
from utils.memory_utils import log_memory_usage, clear_memory

def main():
    parser = argparse.ArgumentParser(description='Run time series mixup experiments')
    parser.add_argument('--config', type=str, default='short_datasets_config.yaml',
                        help='Configuration file name')
    parser.add_argument('--output_dir', type=str, default='/kaggle/working/results',
                        help='Output directory for results')
    args = parser.parse_args()
    
    print("="*60)
    print("Time Series Mixup Benchmark")
    print("="*60)
    
    # 加载配置
    print("\nLoading configuration...")
    config_loader = ConfigLoader('/kaggle/working/configs')
    config = config_loader.load_config(args.config)
    
    # 打印配置信息
    print(f"Datasets: {len(config['datasets'])}")
    print(f"Models: {[m['name'] for m in config['models'] if m['enabled']]}")
    print(f"Augmentations: {[a['name'] for a in config['augmentations'] if a['enabled']]}")
    
    # 记录初始内存
    log_memory_usage("Initial")
    
    # 运行实验
    print("\nStarting experiments...")
    runner = ExperimentRunner(config, args.output_dir)
    results = runner.run_all_experiments()
    
    # 记录最终内存
    log_memory_usage("Final")
    clear_memory()
    
    print(f"\nExperiments completed! Results saved to {args.output_dir}")
    print(f"Total experiments: {len(results)}")

if __name__ == "__main__":
    main()