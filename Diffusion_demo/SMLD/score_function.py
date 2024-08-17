# -*- coding: utf-8 -*-
# @file score_function.py
# @author zhangshilong
# @date 2024/8/13

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.distributions as D


def gaussian_mixture(coefficients):
    pis, mus, sigmas = map(torch.flatten, torch.tensor(coefficients).split(1, dim=1))

    weights = D.Categorical(pis)
    dists = D.Normal(mus, sigmas)
    mix_dist = D.MixtureSameFamily(weights, dists)
    return mix_dist


def plot(dist):
    x = torch.arange(-5, 5, 0.01)
    x.requires_grad_(True)
    log_prob = dist.log_prob(x)
    prob = log_prob.exp()
    log_prob_grad = torch.autograd.grad(log_prob.sum(), x)[0]

    # p(x)
    sns.lineplot(x=x.detach(), y=prob.detach())
    plt.show()

    # ln p(x)
    sns.lineplot(x=x.detach(), y=log_prob.detach())
    plt.show()

    # score
    sns.lineplot(x=x.detach(), y=log_prob_grad.detach())
    plt.hlines(0, -5, 5)
    plt.show()


dist1 = D.Normal(1, 1)
plot(dist1)

dist2 = gaussian_mixture([
    (0.6, 2, 0.5),
    (0.4, -2, 0.2)
])
plot(dist2)
