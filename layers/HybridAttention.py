from turtle import forward
import torch 
import torch.nn as nn
import torch.nn.init as init
from math import sqrt
from torch.nn import functional as F

class MultiChannelMSA(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, 
                    attention_dropout=0.1, output_attention=True):
        super(MultiChannelMSA, self).__init__()

        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
    
    def forward(self, queries, keys, values, attn_mask, tau = None, delta = None):

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
            return (V, A)
        else:
            return (V, None)


class MCMSALayer(nn.Module):

    def __init__(self, configs, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(MCMSALayer, self).__init__()
        self.n_heads = n_heads
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        # self.query_projection = nn.Linear(configs.C_in, d_keys * n_heads, bias=False)
        # self.key_projection = nn.Linear(configs.C_in, d_keys * n_heads, bias=False)
        # self.value_projection = nn.Linear(configs.C_in, d_values * n_heads,bias=False)

        self.W_q = nn.Parameter(torch.empty(configs.C_in, d_keys * n_heads), requires_grad=True)
        self.W_k = nn.Parameter(torch.empty(configs.C_in, d_keys * n_heads), requires_grad=True)
        self.W_v = nn.Parameter(torch.empty(configs.C_in, d_values * n_heads), requires_grad=True)
        # self.out_projection = nn.Linear(d_values,  d_values)

        init.kaiming_normal_(self.W_q, mode='fan_in')
        init.kaiming_normal_(self.W_k, mode='fan_in')
        init.kaiming_normal_(self.W_v, mode='fan_in')

    # def _init_weights(self):
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             init.kaiming_normal__(m.weight)
    #             if m.bias is not None:
    #                 init.zeros_(m.bias)


    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):

        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = queries @ self.W_q
        keys = keys @ self.W_k
        values = values @ self.W_v

        queries = queries.view(B, L, H, -1)
        keys = keys.view(B, S, H, -1)
        values = values.view(B, S, H, -1)


        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        # out = self.out_projection(out) # [B,L,H,D_k]
        return out, attn


class QualityRelatedAttention(nn.Module):
    def __init__(self, configs, output_attention=False):
        super(QualityRelatedAttention, self).__init__()
  
  

        self.output_attention = output_attention


        self.W_k = nn.Parameter(torch.empty(configs.seq_len, configs.seq_len), requires_grad=True)
        init.kaiming_uniform_(self.W_k, mode='fan_in')
    
    def forward(self, queries, keys, values=None):
        """
        queries: Queries of the attention layer [B,T]
        keys: Keys of the attention layer [B,T,d_keys]
        values: Values of the attention layer [B,T,d_values]
        """
        B, T, D = keys.shape


        values =  keys
        keys = keys.permute(0,2,1) @ self.W_k # [B,d_keys,T]

        scale = 1. / sqrt(T+1)

        score =  torch.einsum("bdt,bt->bd", keys, queries) # [B, d_keys]
        
        score = torch.softmax(scale *score, dim=-1)

        score = score.unsqueeze(1).repeat(1,T,1) # [B,T,d_keys]

        out = score * values
        
        if self.output_attention:
            return (out.contiguous(), score)
        else:
            return (out.contiguous(), None)



# class QualityRelatedAttention(nn.Module):
#     def __init__(self,  attention_dropout=0.1, output_attention=True, scale=None):
#         super(QualityRelatedAttention, self).__init__()
#         self.scale = scale
#         self.output_attention = output_attention
#         self.dropout = nn.Dropout(attention_dropout)
    
#     def forward(self, queries, keys, values, attn_mask, tau = None, delta = None):
#         """
#         :param queries: Queries of the attention layer [B,1,T]
#         :param keys: Keys of the attention layer [B,H,D_keys,T]
#         :param values: Values of the attention layer [B,T,H, d_values]
#         :param attn_mask: Masking matrix to prevent attention to certain positions
#         :param tau: Temperature parameter for softmax
#         :param delta: Delta parameter for quality-related attention
#         :return:
#         """
#         B, L, H, E = keys.shape
#         _, S, _, D = values.shape
#         scale = self.scale or 1. / sqrt(E)

#         scores = torch.einsum("bys,bhds>bhyd", queries, keys) # [B,H,1,d]
#         scores = torch.repeat(scores, (1, 1, L, 1)) # [B,H,T,d]
#         # scores = scores.permute(0,2,1,3) # [B,H,T,d]
#         # V: [B,T,H, D_keys]
#         A = self.dropout(torch.softmax(scale * scores, dim=-1))

#         V = torch.einsum("bhld,bthd->blhd", A, values) # [B,T,H,D_k]

#         if self.output_attention:
#             return (V, A)
#         else:
#             return (V, None)

# class QRSAMLayer(nn.Module):
#     def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
#         super(QRSAMLayer, self).__init__()
#         self.inner_attention = attention


#         self.query_projection = nn.Linear(1, d_keys)
#         self.key_projection = nn.Linear(d_keys, d_keys)
#         self.value_projection = nn.Linear(d_keys, d_keys)
#         self.out_projection = nn.Linear(d_keys , d_keys)
#                 # self.out_projection = nn.Linear(d_keys * n_heads, d_model)
    

#     def forward(self,queries, keys, values, attn_mask, tau=None, delta=None):
#         """
#         :param queries: Queries of the attention layer [B,T,1]
#         :param keys: Keys of the attention layer [B,T,H,D_keys]
#         :param values: Values of the attention layer [B,T,H,D_keys]
#         :param attn_mask: Masking matrix to prevent attention to certain positions
#         :param tau: Temperature parameter for softmax
#         :param delta: Delta parameter for quality-related attention
#         :return:
#         """
#         B, L, H, E = keys.shape
#         _, S, _, D = values.shape

#         # queries = self.query_projection(queries) # [B,T,D_keys]
#         # queries = queries.unsqueeze(2) # [B,T,1,D_keys]
#         queries = queries.permute(0,2,1) # [B,1,T]

#         keys = keys.permute(0,2,3,1) # [B,H,D_keys,T]
#         keys = self.key_projection(keys) # [B,H,D_keys,T]

#         values = self.value_projection(values) # [B,T,H, D_keys]


#         out, attn = self.inner_attention(
#             queries,
#             keys,
#             values,
#             attn_mask,
#             tau=tau,
#             delta=delta
#         )
#         # out: [B,T,H,D_keys]
#         out = self.out_projection(out) # [B,T,H,D_keys]
#         return out, attn
    




class DistributedGRUs(nn.Module):
    def __init__(self, configs, d_keys):
        super(DistributedGRUs, self).__init__()
        self.configs = configs
        
        self.GRU_list = nn.ModuleList([
            nn.GRU(d_keys, configs.hidden_dim, configs.e_layers, batch_first=True)
            for _ in range(configs.n_heads)
        ])


    def forward(self,x_enc):
        """
        :param x_enc: [B,H,T,d_keys]
        :return:
        """
        f_list = []
        for i in range(self.configs.n_heads):
            f = x_enc[:, i, :, :]
            gru = self.GRU_list[i]
            f,_ = gru(f) # [B,T,hidden_dim]
            f_list.append(f[:, -1, :]) # [B,hidden_dim]
        f_list = torch.concat(f_list, dim=1) # [B,hidden_dim*H]
        # f_list = f_list.view(f_list.shape[0], -1) # [B,hidden_dim*H]
        return f_list
