import torch

def softmax(values, tau: float = 1.0):
    max_value = torch.max(values, dim=-1, keepdim=True)[0] / tau
    tmp_vals = values / tau

    pref = tmp_vals - max_value
    exp_pref = torch.exp(pref)
    sum_pref = torch.sum(exp_pref, dim=-1).view(-1, 1)
    result = exp_pref / sum_pref
    return result