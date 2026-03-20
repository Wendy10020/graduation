import torch
import torch.nn as nn
from .multi_head_attention import MultiHeadAttention
from .common import point_wise_feed_forward_network

class EncoderLayer(nn.Module):
    """Transformer编码器层"""
    
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        
        self.mha = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        
        self.layernorm1 = nn.LayerNorm(d_model)
        self.layernorm2 = nn.LayerNorm(d_model)
        
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 多头自注意力
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)
        
        # 前馈网络
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2