import pandas as pd
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

class ResultSaver:
    """结果保存器"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建结果文件
        self.results_file = self.output_dir / "experiment_results.csv"
        self.detailed_dir = self.output_dir / "detailed"
        self.detailed_dir.mkdir(exist_ok=True)
        
        # 初始化结果DataFrame
        if not self.results_file.exists():
            self._init_results_file()
    
    def _init_results_file(self):
        """初始化结果文件"""
        df = pd.DataFrame(columns=[
            'timestamp', 'dataset', 'model', 'augmentation', 'mixup_strategy',
            'best_accuracy', 'n_classes', 'n_channels', 'n_timesteps',
            'train_samples', 'test_samples'
        ])
        df.to_csv(self.results_file, index=False)
    
    def save_result(self, result: Dict[str, Any]):
        """保存单个实验结果"""
        # 添加时间戳
        result['timestamp'] = datetime.now().isoformat()
        
        # 提取数据集信息
        dataset_info = result.get('dataset_info', {})
        
        # 准备行数据
        row = {
            'timestamp': result['timestamp'],
            'dataset': result['dataset'],
            'model': result['model'],
            'augmentation': result['augmentation'],
            'mixup_strategy': result.get('mixup_strategy', 'none'),
            'best_accuracy': result['best_accuracy'],
            'n_classes': dataset_info.get('n_classes', 0),
            'n_channels': dataset_info.get('n_channels', 0),
            'n_timesteps': dataset_info.get('n_timesteps', 0),
            'train_samples': dataset_info.get('train_samples', 0),
            'test_samples': dataset_info.get('test_samples', 0)
        }
        
        # 保存到CSV
        df = pd.read_csv(self.results_file)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(self.results_file, index=False)
        
        # 保存详细信息
        detailed_path = self.detailed_dir / f"{result['dataset']}_{result['model']}_{result['augmentation']}_{result.get('mixup_strategy', 'none')}.json"
        with open(detailed_path, 'w') as f:
            json.dump(result, f, indent=2)
    
    def save_summary(self, results: List[Dict[str, Any]]):
        """保存汇总结果"""
        summary = {}
        for result in results:
            key = f"{result['dataset']}_{result['model']}"
            if key not in summary:
                summary[key] = {}
            
            aug_key = f"{result['augmentation']}_{result.get('mixup_strategy', 'none')}"
            summary[key][aug_key] = result['best_accuracy']
        
        # 保存为CSV格式的汇总表
        summary_df = pd.DataFrame.from_dict(summary, orient='index')
        summary_df.to_csv(self.output_dir / "summary.csv")
        
        # 保存为文本格式
        with open(self.output_dir / "summary.txt", 'w') as f:
            f.write("Experiment Results Summary\n")
            f.write("="*60 + "\n\n")
            
            for dataset_model, results_dict in summary.items():
                f.write(f"\n{dataset_model}:\n")
                f.write("-"*40 + "\n")
                for aug, acc in results_dict.items():
                    f.write(f"  {aug}: {acc:.2f}%\n")
    
    def load_results(self) -> pd.DataFrame:
        """加载所有结果"""
        if self.results_file.exists():
            return pd.read_csv(self.results_file)
        return pd.DataFrame()