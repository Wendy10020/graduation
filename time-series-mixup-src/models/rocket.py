import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.pipeline import make_pipeline
from sktime.transformations.panel.rocket import Rocket as RocketTransform

class ROCKET(nn.Module):
    """ROCKET模型"""
    
    def __init__(self, input_shape=None, num_classes=None, num_kernels=10000):
        super().__init__()
        self.num_kernels = num_kernels
        self.rocket = RocketTransform(num_kernels=num_kernels)
        self.classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        self.is_fitted = False
        
    def forward(self, x):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        
        if torch.is_tensor(x):
            x = x.cpu().numpy()
        
        if len(x.shape) == 4:
            x = x.squeeze(1)
        
        features = self.rocket.transform(x)
        predictions = self.classifier.predict(features)
        
        return torch.tensor(predictions)
    
    def fit(self, train_loader):
        X_train = []
        y_train = []
        
        for x, y in train_loader:
            X_train.append(x.cpu().numpy())
            y_train.append(y.cpu().numpy())
        
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        
        if len(X_train.shape) == 4:
            X_train = X_train.squeeze(1)
        
        X_rocket = self.rocket.fit_transform(X_train)
        self.classifier.fit(X_rocket, y_train)
        self.is_fitted = True
    
    def predict(self, x):
        if not self.is_fitted:
            raise RuntimeError("Model not fitted yet. Call fit() first.")
        
        if torch.is_tensor(x):
            x = x.cpu().numpy()
        
        if len(x.shape) == 4:
            x = x.squeeze(1)
        
        features = self.rocket.transform(x)
        predictions = self.classifier.predict(features)
        
        return torch.tensor(predictions)