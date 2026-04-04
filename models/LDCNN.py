import torch
import torch.nn as nn
import torch.nn.functional as F
import math



class PositionalEmbedding(nn.Module):
    """
    Learnable Positional Embedding
    """
    def __init__(self, max_len, d_model):
        super(PositionalEmbedding, self).__init__()
        # learnable postional embedding
        self.pe = nn.Parameter(torch.randn(max_len, d_model))
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        batch_size, seq_len, _ = x.size()
 
        return x + self.pe[:seq_len, :].unsqueeze(0)
    


class SimplifiedTemporalAttention(nn.Module):
    """

    """
    def __init__(self, d_model):
        super(SimplifiedTemporalAttention, self).__init__()

        self.context_vector = nn.Parameter(torch.randn(d_model), requires_grad=True)
   
        
    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.size()
        

        u = torch.matmul(x, self.context_vector) / math.sqrt(d_model)
      
        
        # Softmax 沿时间维度归一化
        attention_weights = F.softmax(u, dim=1)
        
        # 加权求和得到上下文表示 (可选，如果只是为了重加权特征则直接乘)
        # 论文中主要用于增强动态特征的表示，这里我们将权重作用于原特征
        out = x * attention_weights.unsqueeze(-1)
        
        return out
    


class DilatedConvBlock(nn.Module):
    """

    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, padding=False):
        super(DilatedConvBlock, self).__init__()
        
        
        
        if padding:
            padding = ((kernel_size - 1) * dilation) // 2
        
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, 
                              padding=padding, 
                              dilation=dilation)
        else:
            self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,  
                              dilation=dilation)
        


        self.ln = nn.LayerNorm(out_channels)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x: [B, C, L]

        x = self.conv(x)
        # Transpose for LayerNorm: [B, L, C]
        x = x.transpose(1, 2)
        x = self.ln(x)
        x = self.activation(x)

     
        # Transpose back: [B, C, L]
        x = x.transpose(1, 2)
        return x
    


class Model(nn.Module):
    """
    完整的 LDCNN 模型架构
    """
    def __init__(self, config):
        super(Model, self).__init__()
        
        self.input_dim = config.C_in
        self.output_dim = config.C_out
        self.d_model = config.d_model
        self.num_blocks = config.e_layers
        self.task = config.task
        # 1. 输入投影层 (将输入变量映射到 d_model 维度)
        self.input_proj = nn.Linear(self.input_dim, self.d_model)
        max_len = 100
        
        # 2. 位置嵌入
        self.pos_embed = PositionalEmbedding(max_len=max_len, d_model=config.d_model)
        

        self.temporal_attention = SimplifiedTemporalAttention(config.d_model)
        
        dilation_list = [1, 2, 4, 8]

        self.conv_blocks = nn.ModuleList()
        for i in range(self.num_blocks):
            dilation = dilation_list[i]
            in_ch = self.d_model 
            out_ch = self.d_model
            if i == 0:
                padding_flag = True
            else:
                padding_flag = False

            self.conv_blocks.append(DilatedConvBlock(in_ch, out_ch, kernel_size=2, dilation=dilation, padding=padding_flag))
            
        self.dropout = nn.Dropout(config.dropout)
        

        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        # 6. 输出回归层
        self.fc = nn.Linear(self.d_model, self.output_dim)

    def soft_sensor(self, x_enc):
        
        """
        x: 输入张量 [batch_size, seq_len, input_dim]
        """
    
        x = self.input_proj(x_enc)  # [B, L, C_in]
        
   
        x = self.pos_embed(x)   # [B, L, C_in]
        x = self.temporal_attention(x) # 
        x = self.dropout(x) 

        # 转换为 Conv1d 需要的格式 [B, C_in, L]
        x = x.transpose(1, 2)
 
   
        for block in self.conv_blocks:
            x = block(x)        # [B, d_model, 1]
            
 
        x = self.adaptive_pool(x)  # 强行将序列长度池化为 1：[B, d_model, 1]


        x = x.squeeze(-1)       # [B, d_model]
        x = self.dropout(x)

        # Step 6: 输出预测
        out = self.fc(x)        # [B, output_dim]

        return out
        



    def forward(self,x_enc,x_mark_enc,x_dec,x_mark_dec, batch_y, flag='train'):
        
        if self.task == 'short_term_forecasting':
            return self.short_term_forecasting(x_enc, x_mark_enc, x_dec, x_mark_dec)
        elif self.task == 'soft_sensor':
            return self.soft_sensor(x_enc)
        else:
            raise ValueError("task type not supported")
        
       
    
