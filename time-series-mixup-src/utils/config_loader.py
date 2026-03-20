import yaml
import os
from typing import Dict, Any
from pathlib import Path

class ConfigLoader:
    """配置加载器，支持YAML文件和配置继承"""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """加载YAML配置文件"""
        full_path = self.config_dir / config_path
        if not full_path.exists():
            raise FileNotFoundError(f"Config file not found: {full_path}")
            
        with open(full_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # 处理配置继承
        if 'include' in config:
            base_config = self.load_config(config['include'])
            config = self._merge_configs(base_config, config)
            del config['include']
            
        # 处理子配置路径
        self._process_sub_configs(config)
        
        return config
    
    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """递归合并两个配置字典"""
        result = base.copy()
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def _process_sub_configs(self, config: Dict):
        """处理子配置路径"""
        for key, value in config.items():
            if isinstance(value, dict):
                if 'sub_config_path' in value:
                    sub_config = self.load_config(value['sub_config_path'])
                    config[key] = self._merge_configs(sub_config, value)
                    del config[key]['sub_config_path']
                else:
                    self._process_sub_configs(value)
    
    def save_config(self, config: Dict, save_path: str):
        """保存配置到文件"""
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)