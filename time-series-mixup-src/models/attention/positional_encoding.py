import torch
import torch.nn as nn
import numpy as np

def positional_encoding(max_len, d_model):
    """位置编码"""
    position = np.arange(max_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    
    pe = np.zeros((max_len, d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    
    return torch.FloatTensor(pe).unsqueeze(0)

class PositionalEncoding(nn.Module):
    """位置编码层"""
    
    def __init__(self, d_model, max_len=1000):
        super().__init__()
        self.register_buffer('pe', positional_encoding(max_len, d_model))
        
    def forward(self, x):
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x