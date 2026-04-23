"""
Quality Prediction Modeling for Industrial Processes Using Multiscale Attention-Based Convolutional Neural Network, IEEE Transactions on Cybernetics, 2024
Thanks to authors: Xiaofeng Yuan, Lingfeng Huang, Lingjian Ye, etc.
Notice: 
This code is for research purposes only and should not be used for commercial purposes.
                                            **The experiment has not yet achieved optimal performance.**
Since the article did not provide the time step size, and because multi-layer downsampling would lead to a dimension of zero, 
we used padding in the last MaxPooling layer to ensure that the dimension would not become zero. Additionally, the original paper did not make evaluation on the DC experiment; 
to maintain consistency with our code framework, we did not follow their experimental setup. Furthermore, in the DC experiment, we found that the final feature map could not form a 96*1 size. 
We hope that researchers who have read this article can provide valuable suggestions regarding our code!
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiscaleConvLayer(nn.Module):
    """A single Multiscale Convolutional Layer.
    It takes a list of kernel sizes and applies them in parallel.
    """
    def __init__(self, in_channels: int, out_channels_per_group: list, kernel_sizes: list, stride=1):
        """
        Args:
            in_channels: Number of input channels.
            out_channels_per_group: Number of output channels for *each* kernel size group.
            kernel_sizes: List of kernel sizes (e.g., [3, 5, 7]).
            stride: Stride for all convolutions.
        """
        super(MultiscaleConvLayer, self).__init__()
        self.kernel_sizes = kernel_sizes
        self.stride = stride
        self.groups = nn.ModuleList()
        
        for k in kernel_sizes:
            # Calculate padding to keep spatial dimensions consistent (as per paper Eq. 10, 11)
            # PH = PW = (f_m - S) / 2. For odd kernels and stride=1, this works perfectly.
            padding = math.ceil((k - stride) / 2.0)
            conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels_per_group,
                kernel_size=k,
                stride=stride,
                padding=padding,
                bias=False  # Often used with BatchNorm
            )
            self.groups.append(conv)
        
        self.bn = nn.BatchNorm2d(out_channels_per_group * len(kernel_sizes))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Apply each convolution group in parallel
        outputs = []
        for conv in self.groups:
            out = conv(x)
            outputs.append(out)
        
        # Concatenate along the channel dimension
        # If input was (B, C_in, H, W), output is (B, C_out_total, H, W)
        x = torch.cat(outputs, dim=1)
        x = self.bn(x)
        x = self.relu(x)
        return x

class GeneralizedMeanPooling(nn.Module):
    """Generalized Mean Pooling (GeM) as described in the paper.
    It's a learnable pooling that can degrade to avg/max pooling.
    """
    def __init__(self, channels, p=3, eps=1e-6):
        super(GeneralizedMeanPooling, self).__init__()
        self.p_raw = nn.Parameter(torch.ones(channels))
        self.eps = 1e-6

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.view(B, C, -1)

        p = torch.clamp(self.p_raw, min=1, max=30)
        # p = torch.sigmoid(self.p_raw) * 4 + 1   # p ∈ [1,5]
        p = p.view(1, C, 1)
      
        x = x.clamp(min=self.eps)

        log_x = torch.log(x)
        log_sum = torch.logsumexp(p * log_x, dim=-1)
        log_mean = log_sum - math.log(x.size(-1))
        out = torch.exp(log_mean / p.squeeze(-1))
        # p = p.view(1, C, 1).expand(B, -1, -1)
        # max_val, _ = x.max(dim=-1, keepdim=True)
        # scaled = x/ max_val.clamp(min=self.eps)
        # pow_scaled = torch.pow(scaled, p)
        # mean_pow = torch.mean(pow_scaled, dim=-1, keepdim=True)
        # out = max_val * torch.pow(mean_pow, 1 / p)

    
        return out


class ChannelAttentionModule(nn.Module):
    """Channel-wise Attention Module as shown in Figure 3."""
    def __init__(self, num_channels, reduction_ratio=16):
        super(ChannelAttentionModule, self).__init__()
        self.num_channels = num_channels
        self.reduction_ratio = reduction_ratio
        
        # Use GeM for global information aggregation
        self.gem_pool = GeneralizedMeanPooling(channels=num_channels)
        # self.gem_pool = nn.AdaptiveAvgPool2d(1)
        # Fully Connected layers for attention weight generation
        reduced_channels = max(num_channels // reduction_ratio, 1)  # Ensure at least 1 neuron
        self.fc1 = nn.Linear(num_channels, reduced_channels)
        self.fc2 = nn.Linear(reduced_channels, num_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch_size, num_channels, H, W)
        batch_size, C, H, W = x.size()
        
        # 1. Global feature descriptor using GeM
        # gem_out shape: (batch_size, num_channels, 1, 1)
        gem_out = self.gem_pool(x)
        
        # 2. Flatten for FC layers
        # z shape: (batch_size, num_channels)
        z = gem_out.view(batch_size, C)
        
        # 3. Compute attention weights
        # a shape: (batch_size, num_channels)
        a = self.fc1(z)
        a = F.relu(a)
        a = self.fc2(a)
        a = self.sigmoid(a)
        
        # 4. Reshape attention weights to match input
        # a shape: (batch_size, num_channels, 1, 1)
        a = a.view(batch_size, C, 1, 1)
        
        # 5. Apply attention weights (element-wise multiplication)
        # output shape: (batch_size, num_channels, H, W)
        output = x * a
        return output

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.task = config.task
        self.out_ch_per_channel = [5, 8, 10, 12]
        self.kernel_sizes = [3, 5, 7]
        self.multiscale_conv_list = nn.ModuleList()
        self.input_height = config.seq_len
        self.input_width = config.C_in

        self.fc_nums = [324, config.pred_len * config.C_out]

        current_channels = 1
        height = self.input_height
        width = self.input_width
        for i in range(len(self.out_ch_per_channel)):

            self.multiscale_conv_list.append(MultiscaleConvLayer(current_channels, self.out_ch_per_channel[i], self.kernel_sizes, config.stride))
            current_channels = self.out_ch_per_channel[i] * len(self.kernel_sizes)
            self.multiscale_conv_list.append(ChannelAttentionModule(current_channels, config.reduction_ratio))
            if i < len(self.out_ch_per_channel) - 1:
                self.multiscale_conv_list.append(nn.MaxPool2d(kernel_size=2, stride=2))
                height = int(height // 2)
                width = int(width // 2)
            else:
                self.multiscale_conv_list.append(nn.MaxPool2d(kernel_size=2, stride=2,padding=1))
                height = (height + 2* 1- 2) //2 + 1
                width = (width + 2 * 1- 2) //2 + 1     
        width = 1 if width == 0 else width
        self.flattened_features = current_channels * height * width


        self.fc_layers = nn.ModuleList()
        prev_features = self.flattened_features
        for num_neurons in self.fc_nums:
            self.fc_layers.append(nn.Linear(prev_features, num_neurons))
            if num_neurons != self.fc_nums[-1]:  # Add ReLU for all but the last layer
                self.fc_layers.append(nn.ReLU())
            prev_features = num_neurons

    
    def soft_sensor(self, x_enc):
        for layer in self.multiscale_conv_list:
            
            x_enc = layer(x_enc)
            check_nan(x_enc, layer)
        # Flatten for FC layers
        x_enc = torch.flatten(x_enc, 1)
        
        # Pass through FC layers
        for fc_layer in self.fc_layers:
            x_enc = fc_layer(x_enc)
        
        return x_enc

    def short_term_forecasting(self, x_enc):
        for layer in self.multiscale_conv_list:
            x_enc = layer(x_enc)
        
        # Flatten for FC layers
        x_enc = torch.flatten(x_enc, 1)
        
        # Pass through FC layers
        for fc_layer in self.fc_layers:
            x_enc = fc_layer(x_enc)

        x_enc = x_enc.view(-1, self.config.pred_len, self.config.C_out)

        return x_enc
        
    def forward(self,x_enc,x_mark_enc,x_dec,x_mark_dec, batch_y, flag='train'):
        x_enc = x_enc.unsqueeze(1) # [B, 1, T, D]

        if self.task == 'short_term_forecasting':
            return self.short_term_forecasting(x_enc)
        elif self.task == 'soft_sensor':
            return self.soft_sensor(x_enc)
        else:
            raise ValueError("task type not supported")


def check_nan(x, name):
    if torch.isnan(x).any():
        print("NaN in forward:", name)
        raise RuntimeError