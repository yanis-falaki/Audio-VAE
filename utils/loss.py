import torch

def calculate_kl_loss(mean, log_variance):
    kl_loss = -0.5 * torch.sum(1 + log_variance - mean.pow(2) - log_variance.exp(), dim=1).mean()
    return kl_loss