# -*- coding: utf-8 -*-
# @file tweedie_gaussian_mixture.py
# @author zhangshilong
# @date 2024/8/16

from functools import cache

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.distributions as D

device = "cuda" if torch.cuda.is_available() else "cpu"

x_min = theta_min = -15
x_max = theta_max = 15

pis = torch.tensor([0.5, 0.2, 0.3], dtype=torch.float, device=device)
mus = torch.tensor([-7, 6, 10], dtype=torch.float, device=device)
taus = torch.tensor([1, 0.7, 1.3], dtype=torch.float, device=device)
sigma = torch.tensor(1, dtype=torch.float, device=device)

sample_size = 10000


# ===============================================================================


def tensor_tolist(tensor):
    return [round(x, 4) for x in tensor.tolist()]


def gaussian_mixture(_pis, _mus, _sigmas):
    weights = D.Categorical(_pis)
    dists = D.Normal(_mus, _sigmas)
    return D.MixtureSameFamily(weights, dists)


# 似然分布固定为正态分布，若先验分布为高斯混合分布，则目标分布也是高斯混合分布; 后验分布不确定，只能用贝叶斯公式计算
@cache
def get_prior_dist():
    return gaussian_mixture(pis, mus, taus)


def get_likelihood_dist(theta):
    return D.Normal(theta, sigma)


@cache
def get_target_dist():
    target_pis = pis
    target_mus = mus
    target_taus = torch.sqrt(taus ** 2 + sigma ** 2)
    print(f"target_pis={tensor_tolist(target_pis)} "
          f"target_mus={tensor_tolist(target_mus)} "
          f"target_taus={tensor_tolist(target_taus)}\n")

    return gaussian_mixture(target_pis, target_mus, target_taus)


# 通过模拟的方式验证
def verify_target_dist():
    prior_dist = get_prior_dist()
    thetas = prior_dist.sample([sample_size])
    likelihood_dist = get_likelihood_dist(thetas)
    data = likelihood_dist.sample([])
    sns.histplot(data.cpu(), stat="density", binwidth=0.1)


# 通过贝叶斯公式计算, x是给定的条件, thetas是要计算密度的点
def get_posteriori_prob(thetas, x):
    prior_dist = get_prior_dist()
    likelihood_dist = get_likelihood_dist(thetas)
    target_dist = get_target_dist()

    return (prior_dist.log_prob(thetas) + likelihood_dist.log_prob(x) - target_dist.log_prob(x)).exp()


# 猜测后验概率是每部分正态分布的后验概率组合，是错误的
def guess_posteriori_dist(x):
    n = len(x)
    sigma0_2 = sigma ** 2 / n
    avg_x = sum(x) / n

    sigma0_m2 = 1 / sigma0_2
    taus_m2 = 1 / taus ** 2

    sigmas1_2 = 1 / (sigma0_m2 + taus_m2)
    mus1 = sigmas1_2 * (avg_x * sigma0_m2 + mus * taus_m2)

    print(f"mus1={tensor_tolist(mus1)} "
          f"sigmas1_2={tensor_tolist(sigmas1_2)}\n")

    return gaussian_mixture(pis, mus1, torch.sqrt(sigmas1_2))


def target_score(x):
    x = x.clone().requires_grad_(True)
    target_dist = get_target_dist()
    return torch.autograd.grad(target_dist.log_prob(x).sum(), x)[0]


def tweedie(x):
    return x + sigma ** 2 * target_score(x)


def plot_pdf(dist, interval=0.01, label=None):
    x = torch.arange(x_min, x_max, interval, device=device)
    prob = dist.log_prob(x).exp()
    sns.lineplot(x=x.cpu(), y=prob.cpu(), label=label)


def plot_posteriori_pdf(x, interval=0.01):
    start = min(min(x), theta_min)
    end = max(max(x), theta_max)
    thetas = torch.arange(start, end, interval, device=device)

    prob = get_posteriori_prob(thetas, x)
    sns.lineplot(x=thetas.cpu(), y=prob.cpu(), label="posteriori_bayes")


def plot_target_score(interval=0.01):
    x = torch.arange(x_min, x_max, interval, device=device)
    score = target_score(x)
    sns.lineplot(x=x.cpu(), y=score.cpu(), label="target_score")
    plt.hlines(0, x_min, x_max, colors="black")


# ===============================================================================


# 当sigma很小时，先验分布和目标分布应该十分接近
prior_dist = get_prior_dist()
plot_pdf(prior_dist, label="prior")

target_dist = get_target_dist()
plot_pdf(target_dist, label="target")

# 模拟的目标分布应该十分接近计算的目标分布
verify_target_dist()
plt.show()

# ===============================================================================


plot_pdf(prior_dist, label="prior")
plot_pdf(target_dist, label="target")

# 假设有一个观测样本x
x = torch.tensor([0], dtype=torch.float, device=device)
sns.scatterplot(x=x.cpu(), y=0, marker="o", color="purple")

plot_posteriori_pdf(x)

# 猜测错误，与上面用贝叶斯公式计算的曲线不重合
# 当先验分布退化为单峰正态分布时，两条曲线重合，说明贝叶斯公式的确没有错误
# guess_posteriori_dist = guess_posteriori_dist(x)
# plot_pdf(guess_posteriori_dist, label="posteriori_guess")

theta = tweedie(x)
print(f"x={x.item():.4f} -> theta={theta.item():.4f}\n")
sns.scatterplot(x=theta.cpu(), y=0, marker="o", color="red")
plt.show()

# ===============================================================================


plot_target_score()
sns.scatterplot(x=x.cpu(), y=target_score(x).cpu(), marker="o", color="purple")
plt.show()
