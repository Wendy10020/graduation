import torch.nn as nn
from .inception_time import InceptionTime
from .simple_rnn import SimpleRNN
from .simple_mhsa import SimpleMHSA
from .conv_mhsa import ConvMHSA
from .inception_mhsa import InceptionMHSA
from .rocket import ROCKET

class ModelFactory:
    """模型工厂"""
    
    @staticmethod
    def create_model(model_name, dataset_info, **kwargs):
        """
        创建模型
        
        Args:
            model_name: 模型名称
            dataset_info: 数据集信息，包含n_channels, n_timesteps, n_classes
            **kwargs: 模型特定参数
        """
        input_shape = (dataset_info['n_channels'], dataset_info['n_timesteps'])
        num_classes = dataset_info['n_classes']
        
        if model_name == "InceptionTime":
            return InceptionTime(
                input_shape=input_shape,
                num_classes=num_classes,
                nb_filters=kwargs.get('nb_filters', 32),
                use_residual=kwargs.get('use_residual', True),
                use_bottleneck=kwargs.get('use_bottleneck', True),
                depth=kwargs.get('depth', 6),
                kernel_size=kwargs.get('kernel_size', 41),
                dropout=kwargs.get('dropout', 0.1)
            )
        
        elif model_name == "SimpleRNN":
            return SimpleRNN(
                input_shape=input_shape,
                num_classes=num_classes,
                hidden_size=kwargs.get('hidden_size', 256),
                dropout=kwargs.get('dropout', 0.1)
            )
        
        elif model_name == "SimpleMHSA":
            return SimpleMHSA(
                input_shape=input_shape,
                num_classes=num_classes,
                d_model=kwargs.get('d_model', 512),
                num_heads=kwargs.get('num_heads', 8),
                dff=kwargs.get('dff', 512),
                dropout=kwargs.get('dropout', 0.1)
            )
        
        elif model_name == "ConvMHSA":
            return ConvMHSA(
                input_shape=input_shape,
                num_classes=num_classes,
                conv_filters=kwargs.get('conv_filters', [128, 128, 256, 256, 512]),
                kernel_sizes=kwargs.get('kernel_sizes', [7, 7, 3, 7, 7]),
                strides=kwargs.get('strides', [1, 2, 1, 2, 2]),
                d_model=kwargs.get('d_model', 512),
                num_heads=kwargs.get('num_heads', 8),
                dff=kwargs.get('dff', 512),
                dropout=kwargs.get('dropout', 0.1)
            )
        
        elif model_name == "InceptionMHSA":
            return InceptionMHSA(
                input_shape=input_shape,
                num_classes=num_classes,
                nb_filters=kwargs.get('nb_filters', 32),
                use_residual=kwargs.get('use_residual', True),
                use_bottleneck=kwargs.get('use_bottleneck', True),
                d_model=kwargs.get('d_model', 512),
                num_heads=kwargs.get('num_heads', 8),
                dff=kwargs.get('dff', 512),
                dropout=kwargs.get('dropout', 0.1)
            )
        
        elif model_name == "ROCKET":
            return ROCKET(
                num_kernels=kwargs.get('num_kernels', 10000)
            )
        
        else:
            raise ValueError(f"Unknown model: {model_name}")