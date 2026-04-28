import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim, out_dim)
        self.attn = nn.Linear(2 * out_dim, 1)

    def forward(self, h, A):
        # h: (B, C, T)
        B, C, d = h.shape

        Wh = self.W(h)  # (B, M, d)

        Wh_i = Wh.unsqueeze(2).repeat(1, 1, C, 1)
        Wh_j = Wh.unsqueeze(1).repeat(1, C, 1, 1)

        e = self.attn(torch.cat([Wh_i, Wh_j], dim=-1)).squeeze(-1)

        e = e.masked_fill(A == 0, -1e9)
        alpha = F.softmax(e, dim=-1)

        h_prime = torch.matmul(alpha, Wh)

        return F.relu(h_prime)

def graph_norm(adj):
    B, M, _ = adj.shape
    min, _ = torch.min(adj, dim=2, keepdim=True)
    max, _ = torch.max(adj, dim=2, keepdim=True)

    denorm = max - min
    denorm = torch.where(denorm == 0, torch.ones_like(denorm), denorm)

    adj_norm = (adj - min) / denorm
    return adj_norm



def build_graphs(x):
    # x: (B, M, T)
    B, M, T = x.shape
    # Distance
    dist = torch.cdist(x, x)  # (B, M, M)
    Gd = torch.exp(-dist ** 2)
    Gd = graph_norm(Gd)

    # Correlation
    x_centered = x - x.mean(dim=-1, keepdim=True)
    cov = torch.matmul(x_centered, x_centered.transpose(-1, -2))
    std = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1) + 1e-6)
    Gc = torch.abs(cov / (std.unsqueeze(-1) * std.unsqueeze(-2)))
    Gc = graph_norm(Gc)

    # MI 
    x_mean = x.mean(dim=2, keepdim=True)
    x_std = x.std(dim=2, keepdim=True) + 1e-6
    x_norm = (x - x_mean) / x_std

    
   
    r_matrix = torch.matmul(x_norm, x_norm.transpose(1, 2)) / (T - 1)
    r_matrix = torch.clamp(r_matrix, -0.99, 0.99)
    mi_matrix = -0.5 * torch.log(1 - r_matrix**2)
    Gm = graph_norm(mi_matrix)
 


    A = (Gd + Gc + Gm) / 3
    A = (A > 0.6).float()

    return A


class MultiGAT(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.gat1 = GATLayer(in_dim, hidden_dim)
        self.gat2 = GATLayer(hidden_dim, hidden_dim)
        
    def forward(self, x):
        # x: (B, M, T)

        B, M, T = x.shape

        a = build_graphs(x)
        h = self.gat1(x, a)
        h = self.gat2(h, a)

        return h

def compute_stft_batch(x, n_fft=16):
    # x: (B, M, T)
    B, M, T = x.shape



    X = torch.stft(
        x.reshape(B * M, T),
        n_fft=n_fft,
        return_complex=True
    )

    X = torch.abs(X)

    Freq = X.shape[1]

    X = X.reshape(B, M, Freq, -1).mean(-1)  # (B, M, F)

    return X # (B, M, F)


class MRMCFusion(nn.Module):
    def __init__(self, dim, latent_dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, latent_dim)
        self.fc2 = nn.Linear(dim, latent_dim)

    def forward(self, X1, X2, target=None):
        # X1, X2: (B, M, d_model)

        Z1 = torch.sigmoid(self.fc1(X1))
        Z2 = torch.sigmoid(self.fc2(X2))
        Z = torch.cat([Z1, Z2], dim=-1)
        # ===== Min Redundancy =====
        S = torch.exp(-torch.norm(X1 - X2, dim=-1))
        L2 = torch.mean(S.unsqueeze(-1) * (Z1 - Z2) ** 2)



        # ===== Max Correlation =====
        if target is not None:
            target = target  # (B, H, 1)
            Z1 = Z1.unsqueeze(1) # [B, 1, M, Q]
            Z1 = Z1.permute(0, 1, 3, 2) # [B, 1, Q, M]
            R1 = torch.sigmoid(torch.einsum('BHE,BEQM -> BHQM' , target, Z1)) # [B, H, Q, M]
            R1 = torch.sum(R1, dim=1) # [B, Q, M]
            R1 = torch.mean(R1)
            

            Z2 = Z2.unsqueeze(1) # [B, 1, M, Q]
            Z2 = Z2.permute(0, 1, 3, 2) # [B, 1, Q, M]
            R2 = torch.sigmoid(torch.einsum('BHE,BEQM -> BHQM' , target, Z2)) # [B, H, Q, M]
            R2 = torch.sum(R2, dim=1) # [B, Q, M]
            R2 = torch.sum(R2, dim=1) # [B,M]


            L3 = -1 * torch.sum(R1 + R2, dim=-1) # [B]
            L3 = torch.mean(L3)
        else:
            L3 = 0

        

        return Z, L2, L3


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.time_gat = MultiGAT(configs.seq_len, configs.hidden_dim)

        self.n_fft = configs.n_fft

        if configs.n_fft > configs.seq_len:
            raise ValueError('n_fft should not be greater than seq_len')

        freq_dim = self.n_fft // 2 + 1
        self.pred_len = configs.pred_len
        M = configs.C_in # 过程变量节点数
        self.C_out = configs.C_out


        self.task = configs.task


        self.freq_gat = MultiGAT(freq_dim, configs.hidden_dim)
       

        self.fusion = MRMCFusion(configs.hidden_dim , configs.latent_dim)

        self.lstm = nn.LSTM(
            input_size=configs.latent_dim * 2,
            hidden_size=self.pred_len,
            batch_first=True
        )
        self.dropout = nn.Dropout(configs.dropout)
        self.projection = nn.Linear(configs.C_in , self.C_out)

    def short_term_forecasting(self, x_enc, batch_y=None, flag='train'):
        # x: (B, T, M)
        if flag == 'train':
            y = batch_y[:, -self.pred_len:, -self.C_out:]
        else:
            y = None

        x_enc = x_enc.permute(0, 2, 1)  # (B, C, T)

        # ===== Time Stream =====
        h_time = self.time_gat(x_enc)  # (B, C, d_model)
        h_time = h_time.flatten(2)  # (B, C, d_model)

        # ===== Freq Stream =====
        x_freq = compute_stft_batch(x_enc, self.n_fft) # (B, C, F)
        h_freq = self.freq_gat(x_freq)  # (B, C, d_model)
        # h_freq = h_freq.flatten(2)

        # ===== Fusion =====
        Z, L2, L3 = self.fusion(h_time, h_freq, y)
        
        # ===== LSTM =====
        # out, _ = self.lstm(Z[:,-1,:])
        out_list = []
        feature, _ = self.lstm(Z) # (B, C, PRED_LEN)
        feature = self.dropout(feature)
        pred = self.projection(feature.permute(0, 2, 1)) # (B, PRED_LEN, C_out)
        outputs = {
            'y_pred': pred,
            'L1': L2,
            'L2': L3
        }

        return outputs

    def forward(self, x_enc,x_mark_enc,x_dec,x_mark_dec, batch_y,flag = 'train'):
        if self.task == 'short_term_forecasting':
            return self.short_term_forecasting(x_enc,batch_y, flag)
           
        else:
            raise ValueError(f'Invalid task type: {self.task}. Supporting short_term_forecasting')
