"""
实验运行器
"""

import os
import sys
import torch
import numpy as np
from typing import Dict, Any, List, Optional
from pathlib import Path

# 使用绝对导入
from data.uae_manager import UAEDatasetManager
from models.model_factory import ModelFactory
from augmentations.mixup import Mixup
from augmentations.adaptive_mixup import AdaptiveMixup
from augmentations.augmentation_pipeline import create_augmentation_pipeline
from training.trainer import Trainer
from utils.result_saver import ResultSaver
from utils.memory_utils import log_memory_usage, clear_memory


class ExperimentRunner:
    """实验运行器"""
    
    def __init__(self, config: Dict[str, Any], output_dir: str):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.result_saver = ResultSaver(output_dir)
        
    def run_single_experiment(self, dataset_name: str, model_name: str,
                              augmentation_name: str, mixup_strategy: str = None) -> Optional[Dict[str, Any]]:
        """运行单个实验"""
        print(f"\n{'='*60}")
        print(f"Running: {dataset_name} | {model_name} | {augmentation_name} | {mixup_strategy}")
        print(f"{'='*60}")
        
        try:
            # 加载数据集
            print("Loading dataset...")
            dataset = UAEDatasetManager(dataset_name)
            train_loader, test_loader = dataset.get_dataloaders(
                batch_size=self.config['hyperparameters']['batch_size']
            )
            dataset_info = dataset.get_dataset_info()
            
            # 检查内存限制
            if dataset_info['n_timesteps'] > 512 and model_name in ['SimpleRNN', 'SimpleMHSA']:
                print(f"Skipping {model_name} on {dataset_name} (sequence length {dataset_info['n_timesteps']} > 512)")
                return None
            
            # 创建模型
            print(f"Creating model: {model_name}...")
            model_config = self._get_model_config(model_name, dataset_info)
            model = ModelFactory.create_model(model_name, dataset_info, **model_config)
            
            # 创建增强
            print(f"Creating augmentation: {augmentation_name}...")
            augmentation = self._create_augmentation(augmentation_name, mixup_strategy, dataset_info)
            
            # 创建训练器
            trainer = Trainer(
                model, self.config,
                class_statistics={'class_distribution': dataset_info['class_distribution']}
            )
            
            # 训练
            print("Training...")
            results = trainer.train(train_loader, test_loader, augmentation)
            
            # 保存结果
            experiment_result = {
                'dataset': dataset_name,
                'model': model_name,
                'augmentation': augmentation_name,
                'mixup_strategy': mixup_strategy,
                'best_accuracy': results['best_accuracy'],
                'dataset_info': dataset_info,
                'config': self.config
            }
            
            self.result_saver.save_result(experiment_result)
            return experiment_result
            
        except Exception as e:
            print(f"Error in experiment: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_model_config(self, model_name: str, dataset_info: Dict) -> Dict:
        """获取模型配置"""
        for model_cfg in self.config['models']:
            if model_cfg['name'] == model_name:
                base_config = model_cfg.get('config', {})
                break
        else:
            base_config = {}
        
        config = {
            'input_shape': (dataset_info['n_channels'], dataset_info['n_timesteps']),
            'num_classes': dataset_info['n_classes']
        }
        config.update(base_config)
        
        return config
    
    def _create_augmentation(self, aug_name: str, strategy: str = None,
                            dataset_info: Dict = None):
        """创建增强方法"""
        if aug_name == 'standard_mixup':
            if strategy == 'random':
                return Mixup(alpha=1.0, do_prob=0.5)
            elif strategy == 'intra':
                return Mixup(alpha=1.0, do_prob=0.5)  # 同类mixup需要特殊处理
            elif strategy == 'inter':
                return Mixup(alpha=1.0, do_prob=0.5)  # 异类mixup
            else:
                return Mixup(alpha=1.0, do_prob=0.5)
                
        elif aug_name == 'adaptive_mixup':
            class_statistics = {
                'n_classes': dataset_info['n_classes'],
                'class_distribution': dataset_info['class_distribution']
            }
            return AdaptiveMixup(class_statistics, alpha=1.0, do_prob=0.5)
        
        else:
            # 使用增强管道
            aug_config = self._get_augmentation_config(aug_name)
            return create_augmentation_pipeline(
                aug_config,
                dataset_info['n_timesteps'],
                dataset_info['n_channels']
            )
    
    def _get_augmentation_config(self, aug_name: str) -> Dict:
        """获取增强配置"""
        for aug_cfg in self.config['augmentations']:
            if aug_cfg['name'] == aug_name:
                return aug_cfg
        return {}
    
    def run_all_experiments(self) -> List[Dict]:
        """运行所有实验"""
        all_results = []
        
        # 遍历数据集
        for dataset in self.config['datasets']:
            dataset_name = dataset['name']
            
            # 检查长度限制
            if dataset['n_timesteps'] > 512:
                # 只运行InceptionTime
                models_to_run = [m for m in self.config['models'] 
                               if m['name'] == 'InceptionTime' and m.get('enabled', True)]
            else:
                models_to_run = [m for m in self.config['models'] if m.get('enabled', True)]
            
            # 遍历模型
            for model_cfg in models_to_run:
                model_name = model_cfg['name']
                
                # 遍历增强方法
                for aug_cfg in self.config['augmentations']:
                    if not aug_cfg.get('enabled', True):
                        continue
                    
                    aug_name = aug_cfg['name']
                    strategies = aug_cfg.get('strategies', [None])
                    
                    # 遍历策略
                    for strategy in strategies:
                        try:
                            result = self.run_single_experiment(
                                dataset_name, model_name, aug_name, strategy
                            )
                            if result:
                                all_results.append(result)
                                
                            # 清理GPU内存
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                
                        except Exception as e:
                            print(f"Error in experiment: {e}")
                            continue
        
        # 保存汇总结果
        self.result_saver.save_summary(all_results)
        
        return all_results