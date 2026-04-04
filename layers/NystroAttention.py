import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch
from math import sqrt
import numpy as np
import random
from utils.masking import *


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.05, output_attention=True):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        # scores_temp = scores.detach().numpy()
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask._mask, -np.inf)



        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
        
   
class NystroAttention(nn.Module):
    def __init__(self, mask_flag=False, factor=5, scale=None, attention_dropout=0.1, output_attention=True, num_landmarks = 5, init_option = 'modified'):
        super(NystroAttention, self).__init__()
        self.num_landmarks = num_landmarks
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.scale = scale
        self.factor = factor
        self.init_option = init_option
        self.init_option = init_option
        self.output_attention = output_attention
    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)
        
        if self.num_landmarks == L:
            scores = torch.einsum("blhe,bshe->bhls", queries, keys)
            
        else:
            Q_landmarks = queries.reshape(-1,self.num_landmarks,L//self.num_landmarks, H, E).mean(dim=-3)
            K_landmarks = keys.reshape(-1,self.num_landmarks,S//self.num_landmarks, H, D).mean(dim=-3)
            
            kernel_1 = torch.nn.functional.softmax(torch.einsum("blhe,bshe->bhls", queries, K_landmarks), dim=-1)

            kernel_2 = torch.nn.functional.softmax(torch.einsum("blhe,bshe->bhls", Q_landmarks, K_landmarks), dim=-1)

            kernel_3 = torch.nn.functional.softmax(torch.einsum("blhe,bshe->bhls", Q_landmarks, keys), dim=-1)
            
            scores = torch.matmul(torch.matmul(kernel_1,self.iterative_inv(kernel_2)),kernel_3)
        
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask._mask, -np.inf)
        
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)  
    
    def iterative_inv(self, mat, n_iter=6):
     
        """Iterative matrix inversion using the conjugate gradient method.
        Args:
            mat (torch.Tensor): The matrix to be inverted. size: [batch_size, H, m, m]
            n_iter (int): The number of iterations to perform.
        Returns:
            torch.Tensor: The inverted matrix.
        """
        I = torch.eye(mat.size(-1), device = mat.device)
        K = mat
        if self.init_option =='original':
           V = 1 / torch.max(torch.sum(K,dim = -2)) * K.transpose(-1,-2)
        else:
            V = 1 / torch.max(torch.sum(K,dim = -2),dim=-1).values[:,:,None,None] * K.transpose(-1,-2)
            
        
        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(0.25 * V, 13 * I - torch.matmul(KV, 15 * I-torch.matmul(KV, 7  * I - KV)))
        return V


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)
        # attn_temp = attn.detach().numpy()
        return self.out_projection(out), attn