# -*- coding: utf-8 -*-
"""
DLinear model implementation
"""

import torch
import torch.nn as nn

class moving_avg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # Calculate front and end padding dynamically
        front_pad = (self.kernel_size - 1) // 2
        end_pad = self.kernel_size - 1 - front_pad
        
        front = x[:, 0:1, :].repeat(1, front_pad, 1)
        end = x[:, -1:, :].repeat(1, end_pad, 1)
        
        x = torch.cat([front, x, end], dim=1)
        
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        # Extract trend using moving average
        moving_mean = self.moving_avg(x)
        # Remainder is the seasonal component
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    """
    DLinear Model
    """
    def __init__(self, seq_len, pred_len, channels, individual=False, kernel_size=25):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.individual = individual
        
        # Decomposition block
        self.decomp = series_decomp(kernel_size)

        # Linear layers for trend and seasonal components
        if self.individual:
            # If individual=True, each channel (variable) gets its own linear layer
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            # Weights are shared across all channels
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)
            
    def forward(self, x):
        # x expected shape: [Batch, Sequence_Length, Channels]
        
        # 1. Decompose the input
        seasonal_init, trend_init = self.decomp(x)
        
        # 2. Apply linear projections across the temporal dimension
        if self.individual:
            seasonal_output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype, device=x.device)
            trend_output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype, device=x.device)
            for i in range(self.channels):
                seasonal_output[:, :, i] = self.Linear_Seasonal[i](seasonal_init[:, :, i])
                trend_output[:, :, i] = self.Linear_Trend[i](trend_init[:, :, i])
        else:
            # Permute to apply linear layer over the sequence length, then permute back
            seasonal_output = self.Linear_Seasonal(seasonal_init.permute(0, 2, 1)).permute(0, 2, 1)
            trend_output = self.Linear_Trend(trend_init.permute(0, 2, 1)).permute(0, 2, 1)

        # 3. Add the forecasted components back together
        out = seasonal_output + trend_output
        
        # FIX: Slice the output to only return the Target feature (index 0).
        # This forces the output shape to [Batch, Pred_Len, 1], matching the other models!
        return out[:, :, 0:1]