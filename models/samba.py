# -*- coding: utf-8 -*-
"""
SAMBA model implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba import Mamba


class SAMBA(nn.Module):
    def __init__(self, model_args, hidden, inp, out, embed, cheb_k):
        super().__init__()
        self.args = model_args
        
        # Mamba backbone
        self.mam1 = Mamba(model_args, hidden)
        
        # Graph parameters
        self.cheb_k = cheb_k
        self.gamma = nn.Parameter(torch.tensor(1.))
        
        # Learnable adjacency matrix and embedding
        self.adj = nn.Parameter(torch.randn(model_args.vocab_size, embed), requires_grad=True)
        self.embed_w = nn.Parameter(torch.randn(embed, embed), requires_grad=True)
        
        # Chebyshev polynomial weights
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed, cheb_k, inp, out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed, out))
        
        # Output projections
        self.proj = nn.Linear(model_args.vocab_size, 1)
        self.proj_seq = nn.Linear(model_args.seq_in, 1)
    
    def gaussian_kernel_graph(self, E_A, gamma=1.0):
        N = E_A.size(0)
        E_A_expanded = E_A.unsqueeze(0).expand(N, N, -1)
        E_A_T_expanded = E_A.unsqueeze(1).expand(N, N, -1)
        
        distance_matrix = torch.sum((E_A_expanded - E_A_T_expanded)**2, dim=2)
        A = torch.exp(-gamma * distance_matrix)
        
        dr = nn.Dropout(0.35)
        A = F.softmax(A, dim=1)
        
        return dr(A)
    
    def forward(self, input_ids):
        xx = self.mam1(input_ids)
        
        ADJ = self.gaussian_kernel_graph(self.adj, gamma=self.gamma)
        
        # FIX: Dynamic device allocation instead of hardcoded .cuda()
        I = torch.eye(input_ids.size(2), device=input_ids.device)
        
        support_set = [I, ADJ]
        
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * ADJ, support_set[-1]) - support_set[-2])
        
        supports = torch.stack(support_set, dim=0)
        
        weights = torch.einsum('nd,dkio->nkio', self.adj, self.weights_pool) 
        bias = torch.matmul(self.adj, self.bias_pool)
        x_g = torch.einsum("knm,bmc->bknc", supports, xx.permute(0, 2, 1))
        x_g = x_g.permute(0, 2, 1, 3)
        out = torch.einsum('bnki,nkio->bno', x_g, weights) + bias
        
        return self.proj(out.permute(0, 2, 1))