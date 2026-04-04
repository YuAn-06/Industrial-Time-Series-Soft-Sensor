import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Attention Basis Doubly Residual Stacking Neural Network for Multistep-Ahead Prediction in Soft-Sensing Applications， IEEE Sensor Journal, 2025
Thanks to authors: Yuwei Chen, Bocun He, Zhihuan Song, and Chihang Wei
"""


class AttentionBlock(nn.Module):
    def __init__(self, in_channels, seq_len, pred_len, d_model, C_out):
        super(AttentionBlock, self).__init__()
        self.C_out = C_out
        # self.weight_SA = nn.Parameter(torch.randn(in_channels, seq_len), requires_grad=True)
        # self.weight_TA = nn.Parameter(torch.randn(seq_len, in_channels), requires_grad=True)
        # self.bias_TA = nn.Parameter(torch.randn(seq_len), requires_grad=True)
        # self.bias_SA = nn.Parameter(torch.randn(in_channels), requires_grad=True)

        self.linear_SA = nn.Linear(seq_len, in_channels)
        self.linear_TA = nn.Linear(in_channels, seq_len)

        self.weight_3 = nn.Parameter(torch.randn(d_model, seq_len), requires_grad=True)
        self.bias_3 = nn.Parameter(torch.randn(d_model,1), requires_grad=True)

       


        self.weight_4 = nn.Parameter(torch.randn(in_channels, self.C_out), requires_grad=True)
        self.bias_4 = nn.Parameter(torch.randn(1), requires_grad=True)


        self.weight_B = nn.Parameter(torch.randn(seq_len, d_model), requires_grad=True)
        self.weight_F = nn.Parameter(torch.randn(pred_len, d_model), requires_grad=True)
        self.bias_F = nn.Parameter(torch.randn(pred_len,1), requires_grad=True)
        self.bias_B = nn.Parameter(torch.randn(seq_len,1), requires_grad=True)
        self.activation = nn.ReLU()

   

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.linear_SA.weight)
        nn.init.constant_(self.linear_SA.bias, 0.1)
        nn.init.constant_(self.linear_TA.bias, 0.1)
        nn.init.xavier_uniform_(self.linear_TA.weight)
        nn.init.xavier_uniform_(self.weight_3)
        nn.init.constant_(self.bias_3, 0.1)
        nn.init.xavier_uniform_(self.weight_4)
        nn.init.constant_(self.bias_4, 0.1)
        nn.init.xavier_uniform_(self.weight_B)
        nn.init.constant_(self.bias_B, 0.1)
        nn.init.xavier_uniform_(self.weight_F)
        nn.init.constant_(self.bias_F, 0.1)

    def forward(self, x_enc):
        """
        :param x_enc: (B, T, C)
       
        """
        x_mean = torch.mean(x_enc, dim=-1, keepdim=True) # [B, T]
     
        
        x_mean = x_mean.squeeze(-1) # [B, T]

        SA_score =  torch.softmax(self.linear_SA(x_mean), dim=-1) # [B, C]

        H_SA = SA_score.unsqueeze(1) * x_enc # [B, T, C]

        H_SA_mean = torch.mean(H_SA, dim=1, keepdim=False) # [B, C]
  
        TA_score = torch.softmax(self.linear_TA(H_SA_mean), dim=-1) # [B,T]

        H_TA = TA_score.unsqueeze(-1) * H_SA # [B,T,C]



        h_I = self.activation(self.weight_3 @ H_TA + self.bias_3) # [B, D, C]

        h = self.activation(h_I @ self.weight_4+self.bias_4) # [B, D, C_out]

        y_pred_back = self.activation(self.weight_B @ h + self.bias_B) # [B, S, C_out]

        y_pred_front = self.activation(self.weight_F @ h+ self.bias_F) # [B, P, C_out]

      
        return y_pred_back, y_pred_front

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        self.task = config.task
        self.C_out = config.C_out
        if self.task == 'short_term_forecasting':
            self.in_channels = config.C_in
        else:
            self.in_channels = config.C_out + config.C_in
        self.attention_block_list = nn.ModuleList([AttentionBlock(self.in_channels, config.seq_len, config.pred_len, config.d_model, config.C_out) for _ in range(config.e_layers)])


    def short_term_forecasting(self, x_enc):
        
        x_input = x_enc[:,:, :-self.C_out]
        y_input = x_enc[:,:, -self.C_out:]
        y_input_res = y_input
        y_pred_list = []
        for attention_block in self.attention_block_list:
            x_enc = torch.cat((x_input, y_input_res), dim=-1)
            y_pred_back, y_pred_front = attention_block(x_enc)
            y_pred_list.append(y_pred_front)
            y_input_res = y_input - y_pred_back[:, :, -1:]

        y_pred_list = torch.stack(y_pred_list, dim=-1) # [B,T,C,E_layers]

        y_pred = torch.sum(y_pred_list, dim=-1) # [B,T,C]
        return y_pred
    

    def forward(self,x_enc,x_mark_enc,x_dec,x_mark_dec, batch_y, flag='train'):

        if self.task == 'short_term_forecasting':
            dec_out = self.short_term_forecasting(x_enc)
            dec_out = dec_out.repeat(1, 1, self.in_channels)
            return dec_out
        else:
            raise NameError(
                f'Invalid task name \'{self.task} \'. Please Checkout task name')
