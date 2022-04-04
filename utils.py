import torch
import numpy as np
import random


def set_seeds_all(seed=1):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)


def soft_nll_loss(log_probs, target):
    loss = -(target * log_probs).sum(-1).mean(0)
    return loss


def soft_cross_entropy(logits, target):
    log_probs = torch.log_softmax(logits, dim=-1)
    loss = soft_nll_loss(log_probs, target)
    return loss
