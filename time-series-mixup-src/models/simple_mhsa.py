import torch
import torch.nn as nn
from .attention.encoder_layer import EncoderLayer
from .attention.positional_encoding import PositionalEncoding

class SimpleMHSA(nn.Module):
    """简化版多头自注意力模型"""
    
    def __init__(self, input_shape, num_classes, d_model=512, num_heads=8, dff=512, dropout=0.1):
        super().__init__()
        
        if len(input_shape) == 2:
            n_channels, seq_len = input_shape
        else:
            seq_len, n_channels = input_shape
        
        self.input_proj = nn.Conv1d(n_channels, d_model, kernel_size=1)
        self.pos_encoding = PositionalEncoding(d_model, max_len=seq_len)
        
        self.encoder1 = EncoderLayer(d_model, num_heads, dff, dropout)
        self.encoder2 = EncoderLayer(d_model, num_heads, dff, dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.input_proj(x.transpose(1, 2)).transpose(1, 2)
        x = self.pos_encoding(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = x[:, -1, :]
        x = self.dropout(x)
        logits = self.classifier(x)
        
        return logits