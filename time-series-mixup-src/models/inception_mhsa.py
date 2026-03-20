import torch
import torch.nn as nn
from .inception_time import InceptionBlock
from .attention.encoder_layer import EncoderLayer
from .attention.positional_encoding import PositionalEncoding

class InceptionMHSA(nn.Module):
    """Inception + MHSA混合模型"""
    
    def __init__(self, input_shape, num_classes, nb_filters=32, use_residual=True,
                 use_bottleneck=True, d_model=512, num_heads=8, dff=512, dropout=0.1):
        super().__init__()
        
        n_channels, seq_len = input_shape
        
        self.inception1 = InceptionBlock(n_channels, nb_filters, use_bottleneck, depth=3, use_residual=use_residual)
        current_channels = nb_filters * 4
        self.inception2 = InceptionBlock(current_channels, nb_filters, use_bottleneck, depth=3, use_residual=use_residual)
        self.inception3 = InceptionBlock(current_channels, nb_filters, use_bottleneck, depth=3, use_residual=use_residual)
        
        self.proj = nn.Conv1d(current_channels, d_model, kernel_size=1)
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len)
        
        self.encoder1 = EncoderLayer(d_model, num_heads, dff, dropout)
        self.encoder2 = EncoderLayer(d_model, num_heads, dff, dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.inception3(x)
        
        x = self.proj(x)
        x = x.transpose(1, 2)
        x = self.pos_encoding(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        logits = self.classifier(x)
        
        return logits