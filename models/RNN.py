"""
Copyright (C) 2025
@ Name: RNN.py
@ Time: 2025/1/14 10:49
@ Author: YuAn_L
@ Eamil: yuan_l1106@163.com
@ Software: PyCharm
"""

from layers.Embed import *

class RNN(nn.Module):
    def __init__(self, args):
        super(RNN, self).__init__()
        self.configs = args
        self._build_model()

    def _build_model(self):
        self.bid = 2 if self.configs.if_bid else 1
        dropout = self.configs.dropout if self.configs.e_layers > 1 else 0
        # self.embedding = DataEmbedding(self.configs.C_in, self.configs.d_model[0],self.configs.embed,freq=self.configs.freq,dropout=dropout)
        self.embedding = DataEmbedding(self.configs.C_in, self.configs.d_model[0],self.configs.embed,freq=self.configs.freq
                                       ,dropout=dropout) if self.configs.embed_type == 'mixed embed' else nn.Linear(self.configs.C_in,self.configs.d_model[0])
        if self.configs.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(self.configs.d_model[0], self.configs.d_ff, self.configs.e_layers, batch_first=True, dropout=dropout,
                               bidirectional=self.configs.if_bid)
        elif self.configs.rnn_type =='GRU':
            self.rnn = nn.GRU(self.configs.d_model[0], self.configs.d_ff, self.configs.e_layers,
                                batch_first=True, dropout=dropout,bidirectional=self.configs.if_bid)

        # self.linear_projection = nn.Linear(self.configs.d_ff * self.bid, self.configs.pred_len)
        self.linear_projection = nn.Linear(self.configs.d_ff * self.bid, self.configs.pred_len)
        # self.time_projection = nn.Linear(self.configs.seq_len, self.configs.pred_len)
        self.LayerNorm1 = nn.LayerNorm(self.configs.d_ff)
        # self.LayerNorm2 = nn.LayerNorm(self.configs.d_model[0])
        self.model = nn.ModuleList()
        self.model.append(self.rnn)



    def forward(self,x, x_mark=None):

        batch_size, seq_len, c_dim = x.size()

        if self.configs.embed_type == 'mixed embed':
            x = self.embedding(x,x_mark)  # (batch_size, seq_len, d_model)
        else:
            x = self.embedding(x)  # (batch_size, seq_len, d_model)

        for layer in self.model:
            if self.configs.rnn_type == 'LSTM':
                output, (h_n, c_n) = layer(x)
            elif self.configs.rnn_type == 'GRU':
                output, h_n = layer(x)

            h_n = h_n.contiguous().view(self.configs.e_layers, self.bid, batch_size, -1)

            h_n = h_n[-1].transpose(0,1).contiguous().view(batch_size,-1, self.configs.d_ff) # (batch_size, num_layers, d_ff)


        x = self.LayerNorm1(h_n)
        x = x.view(batch_size, -1) # (batch_size, num_layers * d_ff)
        x = self.linear_projection(x) # 

        return x

