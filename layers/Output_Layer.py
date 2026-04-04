import torch.nn as nn
import torch
class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)
    
    
    
    
class Flatten_head(nn.Module):
    def __init__(self, individual, n_vars, flatten_dim, pred_len, dropout):
        super(Flatten_head, self).__init__()
        self.individual = individual
        self.n_vars = n_vars

        
        if self.individual:
            self.linear_list = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linear_list.append(nn.Linear(flatten_dim, pred_len))
                self.dropouts.append(nn.Dropout(dropout))
                
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear_1 = nn.Linear(flatten_dim, flatten_dim)
            self.linear_2 = nn.Linear(flatten_dim, flatten_dim)
            self.linear_3 = nn.Linear(flatten_dim, flatten_dim)
            self.linear_4 = nn.Linear(flatten_dim, pred_len)
            self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        x :[bs, n_vars, patch_num, patch_len]
        """
        if self.individual:
            x_f_list = []
            for i in range(self.n_vars):
                x_f = self.flattens[i](x[:,i])
                x_f = self.linear_list[i](x_f)
                x_f = self.dropouts[i](x_f)
                x_f_list.append(x_f)
            x = torch.stack(x_f_list, dim=1)
        else:
            x = self.flatten(x)
            x = F.relu(self.linear_1(x)) + x
            x = F.relu(self.linear_2(x)) + x
            x = F.relu(self.linear_3(x)) + x
            x = self.linear_4(x)
            x = self.dropout(x)
        return x
