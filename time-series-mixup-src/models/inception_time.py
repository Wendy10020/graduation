import torch
import torch.nn as nn
import torch.nn.functional as F

class InceptionModule(nn.Module):
    """Inception模块"""
    
    def __init__(self, in_channels, nb_filters=32, use_bottleneck=True, kernel_size=41, stride=1):
        super().__init__()
        
        self.use_bottleneck = use_bottleneck
        self.nb_filters = nb_filters
        self.stride = stride
        
        if use_bottleneck and in_channels > 1:
            self.bottleneck = nn.Conv1d(in_channels, 32, kernel_size=1, padding='same', bias=False)
            in_channels = 32
        
        kernel_sizes = [kernel_size // (2 ** i) for i in range(3)]
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, nb_filters, kernel_size=k, stride=stride, padding='same', bias=False)
            for k in kernel_sizes
        ])
        
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=stride, padding=1)
        self.conv_pool = nn.Conv1d(in_channels, nb_filters, kernel_size=1, padding='same', bias=False)
        
        self.bn = nn.BatchNorm1d(nb_filters * 4)
        
    def forward(self, x):
        if self.use_bottleneck and hasattr(self, 'bottleneck'):
            x = self.bottleneck(x)
        
        conv_outputs = [conv(x) for conv in self.convs]
        
        pool = self.maxpool(x)
        pool = self.conv_pool(pool)
        
        conv_outputs.append(pool)
        x = torch.cat(conv_outputs, dim=1)
        
        x = self.bn(x)
        x = F.relu(x)
        
        return x

class ShortcutLayer(nn.Module):
    """残差连接层"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, padding='same', bias=False)
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self, x, shortcut):
        shortcut = self.conv(shortcut)
        shortcut = self.bn(shortcut)
        return F.relu(x + shortcut)

class InceptionBlock(nn.Module):
    """Inception块"""
    
    def __init__(self, in_channels, nb_filters=32, use_bottleneck=True, 
                 kernel_size=41, stride=1, depth=3, use_residual=True):
        super().__init__()
        
        self.use_residual = use_residual
        self.depth = depth
        self.stride = stride
        
        self.inception_layers = nn.ModuleList()
        current_channels = in_channels
        
        for d in range(depth):
            stride_i = stride if d == depth - 1 else 1
            inception = InceptionModule(current_channels, nb_filters, use_bottleneck, kernel_size, stride_i)
            self.inception_layers.append(inception)
            current_channels = nb_filters * 4
        
        if use_residual:
            self.shortcut = ShortcutLayer(in_channels, current_channels, stride)
        
    def forward(self, x):
        shortcut = x
        
        for inception in self.inception_layers:
            x = inception(x)
        
        if self.use_residual:
            x = self.shortcut(x, shortcut)
        
        return x

class InceptionTime(nn.Module):
    """InceptionTime模型"""
    
    def __init__(self, input_shape, num_classes, nb_filters=32, use_residual=True,
                 use_bottleneck=True, depth=6, kernel_size=41, dropout=0.1):
        super().__init__()
        
        n_channels, seq_len = input_shape
        
        self.inception_blocks = nn.ModuleList()
        current_channels = n_channels
        current_filters = nb_filters
        
        for d in range(depth):
            block = InceptionBlock(current_channels, current_filters, use_bottleneck, 
                                   kernel_size, stride=1, depth=3, use_residual=use_residual)
            self.inception_blocks.append(block)
            current_channels = current_filters * 4
            
            if (d + 1) % 3 == 0 and d < depth - 1:
                current_filters = min(current_filters * 2, 256)
        
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(current_channels, num_classes)
        
    def forward(self, x):
        for block in self.inception_blocks:
            x = block(x)
        
        x = self.global_avg_pool(x).squeeze(-1)
        x = self.dropout(x)
        logits = self.classifier(x)
        
        return logits