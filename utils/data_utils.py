# -*- coding: utf-8 -*-
"""
Data loading and preprocessing utilities
"""

import torch
import torch.utils.data
import pandas as pd
import numpy as np


class MinMaxNorm01:
    """Scale data to range [0, 1] per feature"""
    def __init__(self):
        pass
    
    def fit(self, x):
        # FIX: axis=0 ensures it calculates min/max for each column separately
        self.min = x.min(axis=0)
        self.max = x.max(axis=0)
        
        # Prevent division by zero if a column is constant
        self.max = np.where(self.max == self.min, self.max + 1e-8, self.max)
    
    def transform(self, x):
        x = 1.0 * (x - self.min) / (self.max - self.min)
        return x
    
    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)
    
    def inverse_transform(self, x):
        x = x * (self.max - self.min) + self.min
        return x


def data_loader(X, Y, batch_size, shuffle=True, drop_last=True):
    """Create PyTorch DataLoader from tensors"""
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    X, Y = TensorFloat(X), TensorFloat(Y)
    data = torch.utils.data.TensorDataset(X, Y)
    dataloader = torch.utils.data.DataLoader(
        data, 
        batch_size=batch_size,
        shuffle=shuffle, 
        drop_last=drop_last
    )
    return dataloader


def prepare_data(csv_file, window=5, predict=1, test_ratio=0.15, val_ratio=0.05):
    """
    Prepare data for training from CSV file
    """
    X = pd.read_csv(csv_file, index_col="Date", parse_dates=True)
    X.sort_index(ascending=True, inplace=True)

    if "Name" in X.columns:
        del X["Name"]
        
    X["Target_Return"] = X["Price"].pct_change()

    if "Target" in X.columns:
        del X["Target"]
        
    X.dropna(inplace=True)
    
    cols = ["Target_Return"] + [col for col in X.columns if col != "Target_Return"]
    X = X[cols]
    
    data = X.to_numpy()
    
    # ---------------------------------------------------------
    # FIX: PREVENT GLOBAL SCALING LEAKAGE
    # Calculate how many sequences we will have in total
    total_sequences = data.shape[0] - window - predict + 1
    
    test_len = int(test_ratio * total_sequences)
    val_len = int(val_ratio * total_sequences)
    train_len = total_sequences - test_len - val_len
    
    # Isolate the exact rows that will be used to build the training sequences
    train_data_raw = data[:train_len + window]
    
    # Fit the scaler ONLY on the training rows
    mmn = MinMaxNorm01()
    mmn.fit(train_data_raw)
    
    # Transform the entire dataset using only the bounds learned from the past
    dataset = mmn.transform(data)
    # ---------------------------------------------------------
    
    ran = dataset.shape[0]
    i = 0
    X_seq = []
    Y_seq = []
    
    while i + window + predict <= ran:
        X_seq.append(torch.Tensor(dataset[i:i+window, 0:]))
        Y_seq.append(torch.Tensor(dataset[i+window:i+window+predict, 0]))
        i += 1
    
    XX = torch.stack(X_seq, dim=0)
    YY = torch.stack(Y_seq, dim=0)
    YY = YY[:, :, None]
    
    # Because we already calculated the lengths earlier, we can just split the tensors
    X_train = torch.Tensor.float(XX[:train_len, :, :]).cuda()
    Y_train = torch.Tensor.float(YY[:train_len, :, :]).cuda()
    
    X_val = torch.Tensor.float(XX[train_len:train_len+val_len, :, :]).cuda()
    Y_val = torch.Tensor.float(YY[train_len:train_len+val_len, :, :]).cuda()
    
    X_test = torch.Tensor.float(XX[-test_len:, :, :]).cuda()
    Y_test = torch.Tensor.float(YY[-test_len:, :, :]).cuda()
    
    train_loader = data_loader(X_train, Y_train, 64, shuffle=False, drop_last=False)
    val_loader = data_loader(X_val, Y_val, 64, shuffle=False, drop_last=False)
    test_loader = data_loader(X_test, Y_test, 64, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, test_loader, mmn, XX.shape[2]