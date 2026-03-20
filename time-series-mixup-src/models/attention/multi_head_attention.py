import torch
import torch.nn as nn
from .common import scaled_dot_product_attention

class MultiHeadAttention(nn.Module):
    """多头自注意力"""
    
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.depth = d_model // num_heads
        
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.dense = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def split_heads(self, x, batch_size):
        """分割多头"""
        x = x.view(batch_size, -1, self.num_heads, self.depth)
        return x.transpose(1, 2)
    
    def forward(self, v, k, q, mask=None):
        batch_size = q.size(0)
        
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v, mask)
        
        scaled_attention = scaled_attention.transpose(1, 2).contiguous()
        concat_attention = scaled_attention.view(batch_size, -1, self.d_model)
        
        output = self.dense(concat_attention)
        output = self.dropout(output)
        
        return output, attention_weights