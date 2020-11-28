"""
Implemented Adam method to update Alpha in EM algorithm.
"""

__all__ = ["lglh", "lda_linear_newton"]

import torch


def _lda_linear_gradient(alpha: torch.Tensor, gamma: torch.Tensor, lr):
    alpha_ = alpha.clone().detach().requires_grad_(True)
    log_likelihood = lglh(alpha_, gamma)
    log_likelihood.backward()
    step = alpha_.grad
    return lr * step


def lglh(alpha, gamma):
    len_doc = len(gamma)
    alpha_g = len_doc * (torch.lgamma(alpha.sum(0)) - torch.lgamma(alpha)
                         .sum(0))
    gamma_g = torch.sum((alpha - 1) * (torch.digamma(gamma) - torch.digamma(
        gamma.sum(-1)).view(-1, 1)).sum(0))
    return alpha_g + gamma_g


def lda_linear_newton(alpha: torch.Tensor, gamma,
                      tol: float = 1e-4, max_iter: int = 5000,
                      beta1=0.9, beta2=0.999):
    mt = torch.zeros(size=alpha.shape)
    vt = torch.zeros(size=alpha.shape)
    alpha_tol, iter_num = 1, 0
    lr = 1e-1
    if not alpha.requires_grad:
        alpha.requires_grad = True
    while alpha_tol > tol and iter_num <= max_iter:
        step = _lda_linear_gradient(alpha, gamma, 1)
        mt = beta1 * mt + (1 - beta1) * step
        vt = beta2 * vt + (1 - beta2) * torch.pow(step, 2)
        adam_step = lr * mt / (torch.pow(vt, 0.5) + 0.5)
        alpha = alpha + adam_step
        while torch.min(alpha) <= 0:
            alpha -= adam_step
            adam_step *= 0.1
            alpha += adam_step
        alpha.retain_grad = True
        alpha_tol = torch.norm(step)
        iter_num += 1
    return alpha
