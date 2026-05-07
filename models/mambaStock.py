# -*- coding: utf-8 -*-
"""
Stock Mamba model implementation
"""

import torch
import torch.nn as nn
from .mamba import Mamba

class StockMamba(nn.Module):
    def __init__(self, model_args, hidden):
        super().__init__()
        self.args = model_args
        
        # Core Mamba backbone
        self.mamba = Mamba(model_args, hidden)
        
        # FIX: Project from the input feature dimension (vocab_size) to 1,
        # instead of the 'hidden' dimension. This automatically adapts to 
        # 26 features or 27 features based on the dataset!
        self.proj = nn.Linear(model_args.vocab_size, 1)
    
    def forward(self, input_ids):
        # 1. Pass sequence through the Mamba backbone
        x = self.mamba(input_ids)
        
        # 2. Project to desired output dimension
        out = self.proj(x)
        
        return out