# -*- coding: utf-8 -*-
"""
Graph neural network layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class gconv(nn.Module):
    def __init__(self, inp, hid, embed, cheb_k, n):
        super(gconv, self).__init__()
        self.node_num = n
        self.inp = inp
        self.cheb_k = cheb_k
        self.adj = nn.Parameter(torch.randn(n, embed), requires_grad=True)
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed, cheb_k, inp, hid))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed, hid))
    
    def forward(self, x):
        ADJ = F.softmax(F.relu(torch.mm(self.adj, self.adj.transpose(0, 1))), dim=1)
        
        # FIX: Inherit device from ADJ instead of crashing on CPU/MPS
        support_set = [torch.eye(self.node_num, device=ADJ.device), ADJ]
        
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * ADJ, support_set[-1]) - support_set[-2])
        
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', self.adj, self.weights_pool)
        bias = torch.matmul(self.adj, self.bias_pool)
        x_g = torch.einsum("knm,bmc->bknc", supports, x)
        x_g = x_g.permute(0, 2, 1, 3)
        out_6 = torch.einsum('bnki,nkio->bno', x_g, weights) + bias
        
        return out_6

class AVWGCN(nn.Module):
    def __init__(self, dim_in, hid, cheb_k, n):
        super(AVWGCN, self).__init__()
        self.node_num = n
        self.inp = dim_in
        self.cheb_k = cheb_k
        self.node_embeddings = nn.Parameter(torch.randn(n, dim_in, dim_in), requires_grad=True)
        self.weights_pool = nn.Parameter(torch.FloatTensor(cheb_k, n, dim_in, hid))
        self.bias_pool = nn.Parameter(torch.FloatTensor(n, hid))
    
    def forward(self, x):
        supports = F.softmax(F.relu(self.node_embeddings), dim=2)
        
        # FIX: Inherit device from x
        I = torch.eye(self.inp, device=x.device)
        I2 = I[None, :, :].repeat(x.size(1), 1, 1)
        
        support_set = [I2, supports]
        supports = torch.stack(support_set, dim=0)
        x_g = torch.einsum("bnc,kncm->bknm", x, supports)
        x_gconv = torch.einsum('bknm,knmo->bno', x_g, self.weights_pool) + self.bias_pool
        
        return x_gconv