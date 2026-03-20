import torch
import torch.nn as nn
from .attention.encoder_layer import EncoderLayer

class ConvMHSA(nn.Module):
    """卷积+多头自注意力混合模型"""
    
    def __init__(self, input_shape, num_classes, 
                 conv_filters=[128, 128, 256, 256, 512],
                 kernel_sizes=[7, 7, 3, 7, 7],
                 strides=[1, 2, 1, 2, 2],
                 d_model=512, num_heads=8, dff=512, dropout=0.1):
        super().__init__()
        
        n_channels, seq_len = input_shape
        
        self.conv_layers = nn.ModuleList()
        current_channels = n_channels
        
        for i, (filters, kernel, stride) in enumerate(zip(conv_filters, kernel_sizes, strides)):
            self.conv_layers.append(
                nn.Conv1d(current_channels, filters, kernel_size=kernel, stride=stride, padding='same')
            )
            self.conv_layers.append(nn.ReLU())
            current_channels = filters
        
        self.proj = nn.Conv1d(current_channels, d_model, kernel_size=1)
        
        self.encoders = nn.ModuleList([
            EncoderLayer(d_model, num_heads, dff, dropout) for _ in range(4)
        ])
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
        
        x = self.proj(x)
        x = x.transpose(1, 2)
        
        for encoder in self.encoders:
            x = encoder(x)
        
        x = x.transpose(1, 2)
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.dropout(x)
        logits = self.classifier(x)
        
        return logits