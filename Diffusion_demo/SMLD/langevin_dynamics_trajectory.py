# -*- coding: utf-8 -*-
# @file langevin_dynamics_trajectory.py
# @author zhangshilong
# @date 2024/8/17

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.distributions as D
from tqdm import trange

device = "cuda" if torch.cuda.is_available() else "cpu"

# 使用NCSN图3的例子（但改为一维情况）
true_pis = torch.tensor([0.8, 0.2], dtype=torch.float, device=device)
true_mus = torch.tensor([5, -5], dtype=torch.float, device=device)
true_taus = torch.tensor([1, 1], dtype=torch.float, device=device)

epsilon = torch.tensor(0.1, dtype=torch.float, device=device)

T = 500

x_min = -10
x_max = 10


# ===============================================================================


def gaussian_mixture(_pis, _mus, _sigmas):
    weights = D.Categorical(_pis)
    dists = D.Normal(_mus, _sigmas)
    return D.MixtureSameFamily(weights, dists)


def get_true_dist():
    return gaussian_mixture(true_pis, true_mus, true_taus)


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


def langevin_dynamics_step(data, dist, add_noise=True, noise_level=torch.sqrt(epsilon)):
    score = get_score(dist, data)

    with torch.no_grad():
        data += epsilon / 2 * score
        if add_noise:
            data += D.Normal(0., noise_level).sample(data.shape)

    return data


# ===============================================================================

# 真实数据分布
true_dist = get_true_dist()

x0 = torch.tensor([0], dtype=torch.float, device=device)

x = x0.clone().requires_grad_(True)
xs = x0.clone()
for _ in trange(T):
    x = langevin_dynamics_step(x, true_dist, add_noise=True)
    xs = torch.cat([xs, x])

xs = xs.detach()
prob = true_dist.log_prob(xs).exp()
quantiles = xs.quantile(torch.tensor([0.01, 0.5, 0.99], device=device))

plot_pdf(true_dist, label="true")
# 逐个点画的话很耗时
sns.scatterplot(x=xs.cpu(), y=prob.cpu(), marker="o", color="red")
plt.vlines(quantiles.cpu(), -0.05, 0.4, colors="black", linestyles="--")
plt.show()

# x(t)轨迹
sns.lineplot(xs.cpu(), label="trajectory")
plt.hlines(quantiles.cpu(), 0, T, colors="black", linestyles="--")
plt.show()

# score图
plot_score(true_dist, label="true_score")
point_xs = torch.cat([quantiles, x0])
point_ys = get_score(true_dist, point_xs)
sns.scatterplot(x=point_xs.cpu(), y=point_ys.cpu(), marker="o", color="red")

for point_x, point_y in zip(point_xs.tolist(), point_ys.tolist()):
    plt.annotate(f"({point_x:.2f}, {point_y:.2f})", (point_x, point_y))
plt.show()
