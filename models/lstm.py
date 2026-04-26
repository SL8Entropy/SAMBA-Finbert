# -*- coding: utf-8 -*-
"""
Baseline LSTM model for stock price forecasting.
Designed to be compatible with spatial-temporal data loaders and SHAP explainers.
"""

import torch
import torch.nn as nn

class LSTM(nn.Module):
    # UPGRADE: Added dropout=0.2 as a default parameter for better regularization
    def __init__(self, input_size, hidden_size=64, num_layers=2, output_size=1, dropout=0.2):
        """
        Initialize the LSTM baseline model.
        
        Args:
            input_size (int): Number of input features/nodes (num_features).
            hidden_size (int): Number of features in the hidden state.
            num_layers (int): Number of stacked LSTM layers.
            output_size (int): Prediction horizon (how many steps into the future to predict).
            dropout (float): Dropout probability between LSTM layers to prevent overfitting.
        """
        super(LSTM, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size

        # PyTorch LSTM Layer
        # UPGRADE: Added dropout. (Note: PyTorch ignores dropout if num_layers == 1)
        self.lstm = nn.LSTM(
            input_size=input_size, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size, input_size * output_size)

    def forward(self, x):
        """
        Forward pass for the LSTM.
        """
        # 1. Shape Handling
        if x.dim() == 4:
            # Assume shape is (B, 1, N, T) -> Squeeze channels to get (B, N, T)
            x = x.squeeze(1)
            # Permute to get (B, T, N) where N (nodes) acts as our features
            x = x.permute(0, 2, 1)
            
        # 2. LSTM Pass with SHAP Safety
        # SHAP requires gradients while the model is in eval mode. cuDNN crashes when this happens.
        # This dynamically disables cuDNN only when SHAP is probing the model.
        is_shap_probing = (not self.training) and torch.is_grad_enabled()
        
        if is_shap_probing:
            with torch.backends.cudnn.flags(enabled=False):
                out, _ = self.lstm(x)
        else:
            out, _ = self.lstm(x)  

        # 3. Extract the last time step
        last_out = out[:, -1, :]  # shape: (batch_size, hidden_size)

        # 4. Decode to Future Horizon
        pred = self.fc(last_out)  # shape: (batch_size, input_size * output_size)

        # 5. Reshape to match the SAMBA output shape
        pred = pred.view(-1, self.input_size, self.output_size)

        return pred