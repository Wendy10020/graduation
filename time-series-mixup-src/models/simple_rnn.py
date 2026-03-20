import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    """简化版RNN模型"""
    
    def __init__(self, input_shape, num_classes, hidden_size=256, dropout=0.1):
        super().__init__()
        
        n_channels, seq_len = input_shape
        
        self.lstm1 = nn.LSTM(n_channels, hidden_size, batch_first=True, dropout=dropout)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size * 2, batch_first=True, dropout=dropout)
        self.lstm3 = nn.LSTM(hidden_size * 2, hidden_size * 2, batch_first=True, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        x = x.transpose(1, 2)
        
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x, _ = self.lstm3(x)
        
        x = x[:, -1, :]
        x = self.dropout(x)
        logits = self.classifier(x)
        
        return logits