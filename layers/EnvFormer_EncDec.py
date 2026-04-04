import torch 
import torch.nn as nn

from torch.nn import functional as F

class EnvDecomp(nn.Module):
    def __init__(self, kernel_size):
        super(EnvDecomp, self).__init__()
        self.kernel_size = kernel_size
    def linear_interp_vectorized(self,t_idx, max_points, max_values):
        """
        Vectorized linear interpolation for multiple sequences.
        
        Args:
            t_idx: [B, L] — query points (e.g., time indices)
            max_points: [B, K_max] — known x positions (padded)
            max_values: [B, K_max] — known y values (padded)
            mask: [B, K_max] — boolean mask indicating valid points
        
        Returns:
            y_interp: [B, L]
        """
        B, L = t_idx.shape
        _, K = max_points.shape

        # Clamp t_idx to valid range per batch
        t_min = max_points[:, :1]          # [B, 1]
        t_max = max_points[:, -1:]         # [B, 1]
        t_clamped = torch.clamp(t_idx, t_min, t_max)  # [B, L] 防止t_idx 越界

        # Expand for searchsorted: [B, L] vs [B, K]
        idx = torch.searchsorted(max_points, t_clamped, right=True)  # [B, L] 返回第一个大于x_clamped的值
        idx = torch.clamp(idx, 1, K - 1)  # ensure idx in [1, K-1] 防止idx越界

        # Gather neighbors
        t0 = torch.gather(max_points, 1, idx - 1)      # [B, L] 按行 找到插值的左侧位置
        t1 = torch.gather(max_points, 1, idx)          # [B, L] 按行 找到插值的左侧位置
        y0 = torch.gather(max_values, 1, idx - 1)      # [B, L] 按行 找到插值的左侧的值
        y1 = torch.gather(max_values, 1, idx)          # [B, L] 按行 找到插值的左侧的值

        # Linear interpolation: x_t = x_n + (x_m+1-x_n)/(m-n)*(t-n), where n < t <m
        dt = t1 - t0 + 1e-8   
        slope = (y1 - y0) / dt
        y_interp = y0 + slope * (t_clamped - t0)
        return y_interp
    
    
    def compute_envelope(self, signal: torch.Tensor, mask: torch.Tensor, t_idx:torch.Tensor) -> torch.Tensor:
        # signal, mask: [BD, L]
        env_out = torch.zeros_like(signal)
        B, L = t_idx.shape
        BD, _ = signal.shape # BD = B*D
        # For each sequence in BD, get extreme points
        # We'll process all at once using masking and padding
        max_counts = mask.sum(dim=1)  # [BD]
        max_k = max_counts.max().item()

        if max_k == 0:
            return signal.mean(dim=1, keepdim=True).expand(-1, L)

        # Create padded arrays for max_points and max_values
        max_points = torch.full((BD, max_k), L - 1, dtype=torch.float32, device=signal.device)
        max_values = torch.full((BD, max_k), 0.0, dtype=torch.float32, device=signal.device)

        for i in range(BD): # 记录最值的位置
            n = max_counts[i].item()
            if n == 0:
                continue
            idxs = torch.where(mask[i])[0]  # [n]
            max_points[i, :n] = idxs.float()
            max_values[i, :n] = signal[i, idxs]

        # Interpolate
        env_interp = self.linear_interp_vectorized(t_idx, max_points, max_values)  # [BD, L]
        return env_interp


    def forward(self, x: torch.Tensor):
        """
        Input:
            x: [B, L, D]
        Output:
            trend: [B, L, D]
            residual: [B, L, D]
        """
        B, L, D = x.shape
        pad = self.kernel_size // 2
        
        # [B, D, L] for conv1d
        x_t = x.permute(0, 2, 1)
        x_pad = F.pad(x_t, (pad, pad), mode='replicate')  # [B, D, L + 2*pad]

        # Upper envelope: local max; using max_pool1d to search max point and min point, instead of using loop search.
        upper = F.max_pool1d(x_pad, kernel_size=self.kernel_size, stride=1)
        # Lower envelope: local min = -max(-x)
        lower = -F.max_pool1d(-x_pad, kernel_size=self.kernel_size, stride=1)

        # Trim to original length
        upper = upper[:, :, :L].permute(0, 2, 1)  # [B, L, D]
        lower = lower[:, :, :L].permute(0, 2, 1)

        trend = (upper + lower) / 2.0
        residual = x - trend

        return trend, residual


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()

        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        for attn_layer in self.attn_layers:
            x, attn= attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
            attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)
        return x, attns
    

class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, kernel_size, d_ff=None,  dropout=0.1, activation="relu", fft_layer=None):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.envdecomp = EnvDecomp(kernel_size)
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)
        x, _ = self.envdecomp(x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        y, _ = self.envdecomp(y+x)
        
        return self.norm2(y), attn
    



class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection
        self.SEnet = SE_Fusion(num_trends=3)
    def forward(self, x, cross, trend_input, x_mask=None, cross_mask=None, tau=None, delta=None):
        B, _ , _ = x.size()
       
        for layer in self.layers:
            trend_list = []
            x, trend = layer(x, cross, trend_input, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta) # trend:# [B,3,L,D]
            # trend = torch.concat([trend_input, trend], dim=0) # [B,3,L,D]
            trend_list.append(trend) # M \times [B,3,L,D]
            
        feature_trend = self.SEnet(trend_list) 
        
        if self.norm is not None:
            x = self.norm(x)
        
        if self.projection is not None:
            x = self.projection(x)
            x = x + feature_trend
        return x
    
class SE_Fusion(nn.Module):
    def __init__(self, num_trends, reduction=1):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_trends, num_trends // reduction),
            nn.ReLU(),
            nn.Linear(num_trends // reduction, num_trends),
            nn.Sigmoid()
        )

    def forward(self, trends):
        # trends: list of [B, 3, L, D]
        stacked = torch.stack(trends, dim=1)      # [B, M, 3, L, D]
        stacked = stacked.reshape(-1, 3 * stacked.size(1),stacked.size(-2),stacked.size(-1)) # [B, M*3, L, D]
        z = stacked.mean(dim=(2, 3))              # [B, M*3] ← Squeeze
        w = self.fc(z).unsqueeze(-1).unsqueeze(-1)  # [B, M*3, 1, 1] ← Excitation
        fused = (stacked * w).sum(dim=1)          # [B, L, D] ← Weighted sum
        return fused
   
    
class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, kernel_size, C_in, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.envdecomp = EnvDecomp(kernel_size)
        
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=C_in, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
    def forward(self, x, cross, trend_input, x_mask=None, cross_mask=None, tau=None, delta=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask,
            tau=tau, delta=None
        )[0])
        x = self.norm1(x)
        trend_1, x = self.envdecomp(x)
        trend_1 = self.projection(trend_1.permute(0,2,1)).transpose(1,2)
        trend_1 = trend_1 + trend_input
        
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask,
            tau=tau, delta=delta
        )[0])

        y = x = self.norm2(x)
        trend_2, y = self.envdecomp(y)
        trend_2 = self.projection(trend_2.permute(0,2,1)).transpose(1,2)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        y = x + y 
        trend_3, y = self.envdecomp(y)
        trend_3 = self.projection(trend_3.permute(0,2,1)).transpose(1,2)
        trend_list = [trend_1, trend_2,trend_3]
        trend = torch.stack(trend_list, dim=0) # [B,3,L,D]
        return self.norm3(x + y), trend
