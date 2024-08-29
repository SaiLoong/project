# -*- coding: utf-8 -*-
# @file perturb.py
# @author zhangshilong
# @date 2024/8/15

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.distributions as D

device = "cuda" if torch.cuda.is_available() else "cpu"

# 使用NCSN图3的例子（但改为一维情况）
true_pis = torch.tensor([0.8, 0.2], dtype=torch.float, device=device)
true_mus = torch.tensor([5, -5], dtype=torch.float, device=device)
true_taus = torch.tensor([1, 1], dtype=torch.float, device=device)

x_min = -10.
x_max = 10.


# ===============================================================================


def gaussian_mixture(_pis, _mus, _sigmas):
    weights = D.Categorical(_pis)
    dists = D.Normal(_mus, _sigmas)
    return D.MixtureSameFamily(weights, dists)


def get_true_dist():
    return gaussian_mixture(true_pis, true_mus, true_taus)


def get_perturb_dist(_sigma_i):
    perturb_pis = true_pis
    perturb_mus = true_mus
    perturb_taus = torch.sqrt(true_taus ** 2 + _sigma_i ** 2)
    return gaussian_mixture(perturb_pis, perturb_mus, perturb_taus)


def get_score(dist, x):
    requires_grad = x.requires_grad
    x.requires_grad_(True)
    score = torch.autograd.grad(dist.log_prob(x).sum(), x)[0]
    x.requires_grad_(requires_grad)
    return score


def plot_pdf(dist, _x_min=x_min, _x_max=x_max, interval=0.01, range_as=None, label=None):
    if range_as is not None:
        _x_min = min(range_as).item()
        _x_max = max(range_as).item()

    x = torch.arange(_x_min, _x_max, interval, device=device)
    prob = dist.log_prob(x).exp()
    sns.lineplot(x=x.cpu(), y=prob.cpu(), label=label)


def plot_score(dist, _x_min=x_min, _x_max=x_max, interval=0.01, range_as=None, label=None):
    if range_as is not None:
        _x_min = min(range_as).item()
        _x_max = max(range_as).item()

    x = torch.arange(_x_min, _x_max, interval, device=device)
    score = get_score(dist, x)

    sns.lineplot(x=x.cpu(), y=score.cpu(), label=label)

    # ratio = 2 / torch.sqrt(epsilon).item()
    # plt.hlines([-ratio, 0, ratio], _x_min, _x_max, colors="black", linestyles="--")
    plt.hlines(0, _x_min, _x_max, colors="black", linestyles="--")


def plot_score_ratio(dists, _x_min=x_min, _x_max=x_max, interval=0.01, range_as=None):
    if range_as is not None:
        _x_min = min(range_as).item()
        _x_max = max(range_as).item()

    x = torch.arange(_x_min, _x_max, interval, device=device)

    baseline_dist = get_perturb_dist(1)
    baseline_score = get_score(baseline_dist, x)

    avg_ratios = dict()
    for noise_sigma, dist in dists.items():
        score = get_score(dist, x)
        ratio = score / baseline_score
        avg_ratios[noise_sigma] = ratio.mean().item()
        sns.lineplot(x=x.cpu(), y=ratio.cpu(), label=f"perturb {noise_sigma=:}")
    plt.hlines(0, _x_min, _x_max, colors="black", linestyles="--")
    plt.show()

    sns.lineplot(avg_ratios)
    sns.scatterplot(avg_ratios, color="red")
    for noise_sigma, avg_ratio in avg_ratios.items():
        plt.annotate(f"({noise_sigma:.1f}, {avg_ratio:.2f})", (noise_sigma, avg_ratio))

    # 对比曲线，找出sigma的次数
    # sigmas = torch.arange(min(avg_ratios), max(avg_ratios), interval)
    # ys = sigmas ** -1
    # sns.lineplot(x=sigmas, y=ys)
    # plt.show()


# =====================================================================================


# 真实数据分布
true_dist = get_true_dist()

# 扰动数据分布
noise_sigmas = [0.1, 0.2, 0.5, 1, 2, 5, 10]
perturb_dists = {noise_sigma: get_perturb_dist(noise_sigma) for noise_sigma in noise_sigmas}

# 密度曲线对比
plot_pdf(true_dist, label="true")
for noise_sigma, perturb_dist in perturb_dists.items():
    plot_pdf(perturb_dist, label=f"perturb {noise_sigma=:}")
plt.show()

# 分数曲线对比
plot_score(true_dist, label="true")
for noise_sigma, perturb_dist in perturb_dists.items():
    plot_score(perturb_dist, label=f"perturb {noise_sigma=:}")
plt.show()

# 分数随噪声变化的比例
plot_score_ratio(perturb_dists, 0, 4)
