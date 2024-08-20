# -*- coding: utf-8 -*-
# @file langevin_dynamics_distribution.py
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

init_start = torch.tensor(-8, dtype=torch.float, device=device)
init_end = torch.tensor(8, dtype=torch.float, device=device)

epsilon = torch.tensor(0.1, dtype=torch.float, device=device)

T = 100
sample_size = 1280

x_min = -10
x_max = 10


# ===============================================================================


def gaussian_mixture(_pis, _mus, _sigmas):
    weights = D.Categorical(_pis)
    dists = D.Normal(_mus, _sigmas)
    return D.MixtureSameFamily(weights, dists)


def get_true_dist():
    return gaussian_mixture(true_pis, true_mus, true_taus)


def get_init_dist():
    return D.Uniform(init_start, init_end)


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

# 初始分布
init_dist = get_init_dist()
x = init_dist.sample([sample_size]).requires_grad_(True)
plt.title(f"init")
plot_pdf(true_dist, label="true")
sns.histplot(x.detach().cpu(), stat="density", binwidth=0.1)
plt.show()

chunk = int(T / 10)
for t in trange(1, T + 1):
    x = langevin_dynamics_step(x, true_dist, add_noise=True)

    if t % chunk == 0:
        plt.title(f"{t=}")
        plot_pdf(true_dist, label="true")
        sns.histplot(x.detach().cpu(), stat="density", binwidth=0.1)
        plt.show()
