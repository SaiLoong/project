# -*- coding: utf-8 -*-
# @file diffuse.py
# @author zhangshilong
# @date 2024/8/19

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.distributions as D

device = "cuda" if torch.cuda.is_available() else "cpu"

x_min = -15
x_max = 15

pis = torch.tensor([0.2, 0.2, 0.3, 0.1, 0.2], dtype=torch.float, device=device)
mus = torch.tensor([-9, -5, 2, 6, 10], dtype=torch.float, device=device)
taus = torch.tensor([1, 0.7, 1.3, 0.8, 2.2], dtype=torch.float, device=device)

# 只能取(0, 1)
alpha = torch.tensor(0.1, dtype=torch.float, device=device)

sample_size = 10000


# ===============================================================================


def tensor_tolist(tensor):
    return [round(x, 2) for x in tensor.tolist()]


def gaussian(_mu, _sigma):
    _mu = torch.tensor(_mu, dtype=torch.float, device=device)
    _sigma = torch.tensor(_sigma, dtype=torch.float, device=device)
    return D.Normal(_mu, _sigma)


def gaussian_mixture(_pis, _mus, _sigmas):
    weights = D.Categorical(_pis)
    dists = D.Normal(_mus, _sigmas)
    return D.MixtureSameFamily(weights, dists)


def get_x0_dist():
    return gaussian_mixture(pis, mus, taus)


def get_forward_dist(x0):
    return D.Normal(torch.sqrt(alpha) * x0, torch.sqrt(1 - alpha))


def get_xt_dist():
    xt_pis = pis
    xt_mus = torch.sqrt(alpha) * mus
    xt_taus = torch.sqrt(alpha * taus ** 2 + 1 - alpha)
    # print(f"xt_pis={tensor_tolist(xt_pis)} "
    #       f"xt_mus={tensor_tolist(xt_mus)} "
    #       f"xt_taus={tensor_tolist(xt_taus)}\n")

    return gaussian_mixture(xt_pis, xt_mus, xt_taus)


# 通过模拟的方式验证
def verify_xt_dist():
    x0_dist = get_x0_dist()
    x0s = x0_dist.sample([sample_size])
    forward_dist = get_forward_dist(x0s)
    xts = forward_dist.sample([])
    sns.histplot(xts.cpu(), stat="density", binwidth=0.1)


# 通过贝叶斯公式计算
def get_backward_prob(x0s, xt):
    x0_dist = get_x0_dist()
    forward_dist = get_forward_dist(x0s)
    xt_dist = get_xt_dist()

    return (x0_dist.log_prob(x0s) + forward_dist.log_prob(xt) - xt_dist.log_prob(xt)).exp()


def plot_pdf(dist, _x_min=x_min, _x_max=x_max, interval=0.01, range_as=None, label=None):
    if range_as is not None:
        _x_min = min(range_as).item()
        _x_max = max(range_as).item()

    xs = torch.arange(_x_min, _x_max, interval, device=device)
    probs = dist.log_prob(xs).exp()
    sns.lineplot(x=xs.cpu(), y=probs.cpu(), label=label)


def plot_backward_pdf(xt, interval=0.01):
    start = min(min(xt), x_min)
    end = max(max(xt), x_max)
    x0s = torch.arange(start, end, interval, device=device)

    probs = get_backward_prob(x0s, xt)
    sns.lineplot(x=x0s.cpu(), y=probs.cpu(), label="backward_bayes")


# ===============================================================================


x0_dist = get_x0_dist()
plot_pdf(x0_dist, label="x0")

xt_dist = get_xt_dist()
plot_pdf(xt_dist, label="xt")

standard_gaussian_dist = gaussian(0, 1)
plot_pdf(standard_gaussian_dist, label="standard_gaussian")

# 模拟的目标分布应该十分接近计算的目标分布
verify_xt_dist()
plt.show()
# alpha=0时，xt就是标准正态分布；alpha->1时，xt近似x0


# ===============================================================================


plot_pdf(x0_dist, label="x0")
plot_pdf(xt_dist, label="xt")

# 假设有一个观测样本x
x = torch.tensor([0], dtype=torch.float, device=device)
sns.scatterplot(x=x.cpu(), y=0, color="red")

plot_backward_pdf(x)
plt.show()
