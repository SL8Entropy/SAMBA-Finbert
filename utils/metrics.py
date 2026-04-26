# -*- coding: utf-8 -*-
"""
Evaluation metrics for traffic prediction
"""

import torch
import numpy as np


def MAE_torch(pred, true, mask_value=None):
    """Mean Absolute Error"""
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true - pred))


def MSE_torch(pred, true, mask_value=None):
    """Mean Squared Error"""
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean((pred - true) ** 2)


def RMSE_torch(pred, true, mask_value=None):
    """Root Mean Squared Error"""
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.mean((pred - true) ** 2))


def RRSE_torch(pred, true, mask_value=None):
    """Root Relative Squared Error"""
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.sqrt(torch.sum((pred - true) ** 2)) / torch.sqrt(torch.sum((pred - true.mean()) ** 2))


def MAPE_torch(pred, true, mask_value=None):
    """Mean Absolute Percentage Error"""
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(torch.div((true - pred), true)))


def PNBI_torch(pred, true, mask_value=None):
    """Percentage of Normalized Bias"""
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    indicator = torch.gt(pred - true, 0).float()
    return indicator.mean()


def oPNBI_torch(pred, true, mask_value=None):
    """Overall Percentage of Normalized Bias"""
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    bias = (true + pred) / (2 * true)
    return bias.mean()


def MARE_torch(pred, true, mask_value=None):
    """Mean Absolute Relative Error"""
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.div(torch.sum(torch.abs((true - pred))), torch.sum(true))


def SMAPE_torch(pred, true, mask_value=None):
    """Symmetric Mean Absolute Percentage Error"""
    if mask_value is not None:
        mask = torch.gt(true, mask_value)
        pred = torch.masked_select(pred, mask)
        true = torch.masked_select(true, mask)
    return torch.mean(torch.abs(true - pred) / (torch.abs(true) + torch.abs(pred)))


def All_Metrics(pred, true, mask1, mask2):
    """Compute all metrics"""
    assert type(pred) == type(true)
    
    if type(pred) == torch.Tensor:
        mae = MAE_torch(pred, true, mask1)
        rmse = RMSE_torch(pred, true, mask1)
        rrse = RRSE_torch(pred, true, mask1)
    else:
        raise TypeError
    
    return mae, rmse, rrse


def pearson_correlation(x, y):
    x = x.float().squeeze()
    y = y.float().squeeze()
    
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    dev_x = x - mean_x
    dev_y = y - mean_y
    
    cov = torch.mean(dev_x * dev_y)
    
    # FIX: Set unbiased=False so standard deviation divides by N instead of N-1
    std_x = torch.std(x, unbiased=False)
    std_y = torch.std(y, unbiased=False)
    
    # Add epsilon to prevent division by zero NaN errors
    return cov / (std_x * std_y + 1e-8)


def rank_tensor(x):
    """Safe rank mapping"""
    x = x.squeeze()
    sorted_indices = torch.argsort(x)
    ranks = torch.zeros_like(sorted_indices, dtype=torch.float)
    ranks[sorted_indices] = torch.arange(1, len(x) + 1, device=x.device).float()
    return ranks


def rank_information_coefficient(x, y):
    """Safe Rank Information Coefficient"""
    rank_x = rank_tensor(x)
    rank_y = rank_tensor(y)
    
    return pearson_correlation(rank_x, rank_y)