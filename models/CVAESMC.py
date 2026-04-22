import torch
import numpy as np
from torch import nn



"""
A Novel CVAE-Based Sequential Monte Carlo Framework for Dynamic Soft Sensor Applications
Wenxin Sun, Weili Xiong, et al. IEEE TII 2023

"""

def reparameterize(mu, logvar):
    std = torch.exp(logvar * 0.5)
    eps = torch.randn_like(std)
    return mu + eps * std

def monte_carlo_sampling( mu, logvar, num_samples):

    std = torch.exp(logvar * 0.5)
    dist = torch.distributions.Normal(mu, std)
    samples = dist.rsample((num_samples,))

    
    return samples # [  num_samples, batch_size, T_dim, z_dim]

class Encoder(nn.Module):
    def __init__(self, C_in, C_out, T_dim, z_dim, hidden_dim, activation, num_samples):
        super(Encoder, self).__init__()

        input_dim = 4 * (C_in - C_out) + 5 * C_out + C_out
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim//2)
        self.fc_mean_var = nn.Linear(hidden_dim//2, z_dim * 2)
        try:
            self.activation = getattr(nn, activation)()
        except:
            raise NameError(f'Invalid activation name \'{activation} \'. Please Checkout activation name')
        self.num_samples = num_samples
    


    

        
    def forward(self, x_enc):
        """
        x_enc: [B, T * (C_x + C_y)]
        """

        
        x_enc = self.activation(self.fc1(x_enc))
        x_enc = self.activation(self.fc2(x_enc))
        x_enc = self.activation(self.fc_mean_var(x_enc))
        mu_posterior, logvar_posterior = torch.chunk(x_enc, 2, dim=-1) # [batch_size, z_dim]
 
        z_posterior = monte_carlo_sampling(mu_posterior, logvar_posterior, self.num_samples)

        return z_posterior, mu_posterior, logvar_posterior


class Decoder(nn.Module):
    def __init__(self, C_in, C_out, T_dim, z_dim, hidden_dim, num_samples, activation):
        super(Decoder, self).__init__()
        input_dim = 4 * (C_in - C_out) + 5 * C_out 
        self.fc1 = nn.Linear(input_dim + z_dim, C_out )

        self.log_sigma_w = nn.Parameter(torch.zeros((1,1)),requires_grad=True)  # 初始化为0，对应 sigma_w=1
        self.num_samples = num_samples
        self.C_out = C_out
        self.C_in = C_in

        self.w_t = nn.Parameter(torch.zeros(C_out), requires_grad=True)
        try:
            self.activation = getattr(nn, activation)()
        except:
            raise NameError(f'Invalid activation name \'{activation} \'. Please Checkout activation name')
    

    

    def forward(self, z_t, x_t):
        """
        z_t: latent variable [B, n_samples,  C_z]
        x_t: current input yt [B,T * (C_x+C_y)]
        return y_t_1
        """
        B,  _ = x_t.shape


        x_t = x_t.unsqueeze(0).expand(self.num_samples, -1, -1)
        x_t = x_t.reshape(B * self.num_samples, -1) # [B*n_samples, C_x]

        z_t = z_t.reshape(B * self.num_samples, z_t.shape[-1])

        x_dec = torch.cat([x_t, z_t], dim=-1) # [B*n_samples,  C_z+C_x]

        mean_dec = self.fc1(x_dec)
        logvar_dec = self.log_sigma_w



        mean_dec = mean_dec.reshape(self.num_samples, B,  self.C_out) # [n_samples ,B,  C_y]


        return  mean_dec, logvar_dec




class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.C_out = configs.C_out
        self.z_dim = configs.z_dim
        self.encoder = Encoder(configs.C_in, configs.C_out, configs.seq_len, configs.z_dim, configs.hidden_dim, configs.activation, configs.num_samples)
        self.decoder = Decoder(configs.C_in, configs.C_out, configs.seq_len, configs.z_dim, configs.hidden_dim, configs.num_samples, configs.activation)
        self.output_type = configs.output_type
    
    @torch.no_grad()
    def generate(self, dec_inp, num_samples):

        B,  C = dec_inp.shape
        
        z_prior = torch.randn(num_samples, B, self.z_dim, device=dec_inp.device) 

        mean_dec, logvar_dec = self.decoder(z_prior, dec_inp)
        
        sigma_w = torch.exp(logvar_dec)
        omega = torch.randn(num_samples, B, self.C_out, device=dec_inp.device)
        y_pred = mean_dec + sigma_w * omega


        if self.output_type == 'mean':
            return y_pred.mean(dim=0)
        elif self.output_type == 'median':
            return y_pred.median(dim=0).values



    def short_term_forecasting(self, x_enc, dec_inp):
        

        z_posterior, mu_posterior, logvar_posterior = self.encoder(x_enc)

        mean_dec, logvar_dec = self.decoder(z_posterior, dec_inp)

        dec_out = {
            'mu_posterior': mu_posterior, # [B,  C_out]
            'logvar_posterior': logvar_posterior, # [B, C_out]
            'mean_dec': mean_dec, # [B, n_samples, C_out]
            'logvar_dec': logvar_dec # [B,n_samples, C_out]
        }
        return dec_out
    
    def soft_sensor(self, x_enc):
        pass



    def forward(self,x_enc,x_mark_enc,x_dec,x_mark_dec, batch_y, flag='train'):
        """
        x_enc: [B, T, D_X + D_Y]
        batch_y: [B,1, D_Y]
        """
        B, T, _ = x_enc.size()


        u_t = x_enc[:, -4:, :-self.C_out]
        y_t = x_enc[:, :, -self.C_out:]
        u_t = u_t.reshape(B, -1)
        y_t = y_t.reshape(B, -1)

        y_t_1 = batch_y[:, -1:,-self.C_out:] # [B, 1, C_y] for y_t+1
        y_t_1 = y_t_1.reshape(B, -1)
        x_t = torch.cat([u_t, y_t], dim=-1)
        x_enc = torch.cat([x_t, y_t_1], dim=-1)

        dec_inp = x_t



        if self.configs.task == 'soft_sensor':
            raise ValueError('Only support short_term_forecasting task for now')

        elif self.configs.task == 'short_term_forecasting':
            if flag == 'train':
                return self.short_term_forecasting(x_enc,dec_inp)
            else:
                with torch.no_grad():
                    return self.generate(dec_inp, self.configs.num_samples)


