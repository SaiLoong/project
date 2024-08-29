# -*- coding: utf-8 -*-
# @file score.py
# @author zhangshilong
# @date 2024/8/18

import torch
import torch.distributions as D

device = "cuda" if torch.cuda.is_available() else "cpu"


def gaussian_mixture(_pis, _mus, _sigmas):
    weights = D.Categorical(_pis)
    dists = D.Normal(_mus, _sigmas)
    return D.MixtureSameFamily(weights, dists)


def get_score(_dist, _x):
    requires_grad = _x.requires_grad
    _x.requires_grad_(True)
    score = torch.autograd.grad(_dist.log_prob(_x).sum(), _x)[0]
    _x.requires_grad_(requires_grad)
    return score


# ===============================================================================

true_pis = torch.tensor([0.8, 0.2], dtype=torch.float, device=device)
true_mus = torch.tensor([5, -5], dtype=torch.float, device=device)
true_taus = torch.tensor([1, 1], dtype=torch.float, device=device)

sample_size = 10000

dist = gaussian_mixture(true_pis, true_mus, true_taus)
data = dist.sample([sample_size]).requires_grad_(True)
scores = get_score(dist, data)
mean = scores.mean()

# 分数的期望是0
print(f"{mean=}")
