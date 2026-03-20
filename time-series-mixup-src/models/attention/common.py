import torch
import torch.nn as nn
import math

def scaled_dot_product_attention(q, k, v, mask=None):
    """
    缩放点积注意力
    Args:
        q: query [batch, heads, seq_len, depth]
        k: key [batch, heads, seq_len, depth]  
        v: value [batch, heads, seq_len, depth]
        mask: attention mask [batch, 1, seq_len, seq_len]
    Returns:
        output: [batch, heads, seq_len, depth]
        attention_weights: [batch, heads, seq_len, seq_len]
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    
    attention_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attention_weights, v)
    
    return output, attention_weights

def point_wise_feed_forward_network(d_model, dff):
    """点前馈网络"""
    return nn.Sequential(
        nn.Linear(d_model, dff),
        nn.ReLU(),
        nn.Linear(dff, d_model)
    )