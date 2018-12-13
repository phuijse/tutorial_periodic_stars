import torch
import numpy as np
from torch.nn import functional as F
import matplotlib.pylab as plt

def logsumexp(inputs, dim=None, keepdim=True):    
    # From: https://github.com/YosefLab/scVI/issues/13
    return (inputs - F.log_softmax(inputs, dim=dim)).sum(dim, keepdim=keepdim)

class VAE(torch.nn.Module):
    def __init__(self, n_input=40, n_hidden=64, n_latent=2, importance_sampling=False):
        super(VAE, self).__init__()
        self.importance = importance_sampling
        # Encoder layers
        self.enc_hidden = torch.nn.Linear(n_input, n_hidden)
        self.enc_mu = torch.nn.Linear(n_hidden, n_latent)
        self.enc_logvar = torch.nn.Linear(n_hidden, n_latent)
        # decoder layers
        self.dec_hidden = torch.nn.Linear(n_latent, n_hidden) 
        # Experiments with convolutional decoder
        self.dec_mu = torch.nn.Linear(n_hidden, n_input)
        self.dec_logvar = torch.nn.Linear(n_hidden, 1)
        
    def encode(self, x):
        h = F.relu(self.enc_hidden(x))
        return self.enc_mu(h), self.enc_logvar(h)

    def sample(self, mu, logvar, k=1):
        batch_size, n_latent = logvar.shape
        std = (0.5*logvar).exp()
        eps = torch.randn(batch_size, k, n_latent, device=std.device, requires_grad=False)
        return eps.mul(std.unsqueeze(1)).add(mu.unsqueeze(1))

    def decode(self, z):
        h = F.relu(self.dec_hidden(z))
        hatx = self.dec_mu(h)
        return hatx, (self.dec_logvar(h))  
        

    def forward(self, x, k=1):
        enc_mu, enc_logvar = self.encode(x)
        z = self.sample(enc_mu, enc_logvar, k)
        dec_mu, dec_logvar = self.decode(z)
        return dec_mu, dec_logvar, enc_mu, enc_logvar, z
    
    def ELBO(self, x, dec_mu, dec_logvar, enc_mu, enc_logvar, z):   
        logpxz = -0.5*(dec_logvar + (x - dec_mu).pow(2)/dec_logvar.exp()).sum(dim=-1)
    
        mc_samples = z.shape[1]        
        if self.importance: # Importance-Weighted autoencoder (IWAE)
            logqzxpz = 0.5 * (z.pow(2) - z.sub(enc_mu.unsqueeze(1)).pow(2)/enc_logvar.unsqueeze(1).exp() - enc_logvar.unsqueeze(1)).sum(dim=-1)
        else:  # Variational autoencoder
            logqzxpz = -0.5 * (1.0 + enc_logvar - enc_mu.pow(2) - enc_logvar.exp()).sum(dim=-1).unsqueeze_(1)
        ELBO = torch.sum(logsumexp(logqzxpz - logpxz, dim=1) + np.log(mc_samples))
        return ELBO, logpxz.sum()/mc_samples, logqzxpz.sum()/logqzxpz.shape[1]
    
