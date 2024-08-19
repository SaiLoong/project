# -*- coding: utf-8 -*-
# @file annealed_langevin_dynamics.py
# @author zhangshilong
# @date 2024/8/17

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.distributions as D
from tqdm import trange

device = "cuda" if torch.cuda.is_available() else "cpu"

# 使用NCSN图3的例子（但改为一维情况），调整了部分超参数使得最后的扰动分布更接近真实分布
true_pis = torch.tensor([0.8, 0.2], dtype=torch.float, device=device)
true_mus = torch.tensor([5, -5], dtype=torch.float, device=device)
true_taus = torch.tensor([1, 1], dtype=torch.float, device=device)

init_start = torch.tensor(-8, dtype=torch.float, device=device)
init_end = torch.tensor(8, dtype=torch.float, device=device)

sample_size = 1280
L = 10
T = 100

sigma_1 = torch.tensor(2, dtype=torch.float, device=device)
sigma_L = torch.tensor(0.1, dtype=torch.float, device=device)
sigma_factor = (sigma_L / sigma_1) ** (1 / (L - 1))
epsilon = torch.tensor(0.01, dtype=torch.float, device=device)

x_min = -10
x_max = 10


def sigma(i):
    return sigma_1 * sigma_factor ** (i - 1)


# ===============================================================================


def gaussian_mixture(_pis, _mus, _sigmas):
    weights = D.Categorical(_pis)
    dists = D.Normal(_mus, _sigmas)
    return D.MixtureSameFamily(weights, dists)


def get_true_dist():
    return gaussian_mixture(true_pis, true_mus, true_taus)


def get_init_dist():
    return D.Uniform(init_start, init_end)


def get_perturb_dist(_sigma_i):
    perturb_pis = true_pis
    perturb_mus = true_mus
    perturb_taus = torch.sqrt(true_taus ** 2 + _sigma_i ** 2)
    return gaussian_mixture(perturb_pis, perturb_mus, perturb_taus)


def plot_pdf(dist, _x_min=x_min, _x_max=x_max, interval=0.01, range_as=None, label=None):
    if range_as is not None:
        _x_min = min(range_as).item()
        _x_max = max(range_as).item()

    x = torch.arange(_x_min, _x_max, interval, device=device)
    prob = dist.log_prob(x).exp()
    sns.lineplot(x=x.cpu(), y=prob.cpu(), label=label)


# ===============================================================================


def langevin_dynamics_step(_data, dist, _epsilon):
    score = torch.autograd.grad(dist.log_prob(_data).sum(), _data)[0]

    with torch.no_grad():
        _data += _epsilon / 2 * score + D.Normal(0., torch.sqrt(_epsilon)).sample(_data.shape)

    return _data


def langevin_dynamics(_data, dist, _epsilon):
    for _ in trange(T):
        _data = langevin_dynamics_step(_data, dist, _epsilon)
    return _data


# ===============================================================================

# 真实数据分布
true_dist = get_true_dist()

# 初始分布
init_dist = get_init_dist()
data = init_dist.sample([sample_size]).requires_grad_(True)
plt.title("init")
plot_pdf(true_dist, label="true")
sns.histplot(data.detach().cpu(), stat="density", binwidth=0.1)
plt.show()

for i in range(1, L + 1):
    sigma_i = sigma(i)
    alpha_i = epsilon * sigma_i ** 2 / sigma_L ** 2
    print(f"sigma_{i}={sigma_i:.2f} alpha_{i}={alpha_i:.2f}")

    perturb_dist = get_perturb_dist(sigma_i)
    data = langevin_dynamics(data, perturb_dist, alpha_i)

    plt.title(f"sigma_{i}={sigma_i:.2f}      alpha_{i}={alpha_i:.2f}")
    plot_pdf(true_dist, label="true")
    plot_pdf(perturb_dist, range_as=data, label="perturb")
    sns.histplot(data.detach().cpu(), stat="density", binwidth=0.1)
    plt.show()

# 特别地，令sigma_1和sigma_L都很小，相当于几乎不加噪，退化为langevin_dynamics
