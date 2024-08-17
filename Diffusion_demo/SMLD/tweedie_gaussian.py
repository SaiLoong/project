# -*- coding: utf-8 -*-
# @file tweedie_gaussian.py
# @author zhangshilong
# @date 2024/8/16

from functools import cache

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.distributions as D

device = "cuda" if torch.cuda.is_available() else "cpu"

x_min = theta_min = -10
x_max = theta_max = 10

mu = torch.tensor(-3, dtype=torch.float, device=device)
tau = torch.tensor(1.5, dtype=torch.float, device=device)
sigma = torch.tensor(2, dtype=torch.float, device=device)

sample_size = 10000


# ===============================================================================


# 似然分布固定为正态分布，若先验分布为正态分布，则目标分布、后验分布也是正态分布
@cache
def get_prior_dist():
    return D.Normal(mu, tau)


def get_likelihood_dist(theta):
    return D.Normal(theta, sigma)


@cache
def get_target_dist():
    target_mu = mu
    target_sigma = torch.sqrt(tau ** 2 + sigma ** 2)
    print(f"{target_mu=:.4f} {target_sigma=:.4f}\n")

    return D.Normal(target_mu, target_sigma)


# 通过模拟的方式验证
def verify_target_dist():
    prior_dist = get_prior_dist()
    thetas = prior_dist.sample([sample_size])
    likelihood_dist = get_likelihood_dist(thetas)
    data = likelihood_dist.sample([])
    sns.histplot(data.cpu(), stat="density", binwidth=0.1)


def get_posteriori_dist(x):
    n = len(x)
    sigma0_2 = sigma ** 2 / n
    avg_x = sum(x) / n

    sigma0_m2 = 1 / sigma0_2
    tau_m2 = 1 / tau ** 2

    sigma1_2 = 1 / (sigma0_m2 + tau_m2)
    mu1 = sigma1_2 * (avg_x * sigma0_m2 + mu * tau_m2)
    sigma1 = torch.sqrt(sigma1_2)

    print(f"mu1={mu1.item():.4f} sigma1={sigma1.item():.4f}\n")

    return D.Normal(mu1, sigma1)


# 通过贝叶斯公式验证
def verify_posteriori_dist(x, interval=0.01, label=None):
    start = min(min(x), theta_min)
    end = max(max(x), theta_max)
    thetas = torch.arange(start, end, interval, device=device)

    prior_dist = get_prior_dist()
    likelihood_dist = get_likelihood_dist(thetas)
    target_dist = get_target_dist()

    prob = (prior_dist.log_prob(thetas) + likelihood_dist.log_prob(x) - target_dist.log_prob(x)).exp()
    sns.lineplot(x=thetas.cpu(), y=prob.cpu(), label=label)


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
x = torch.tensor([5], dtype=torch.float, device=device)
sns.scatterplot(x=x.cpu(), y=0, marker="o", color="purple")

# 两条后验曲线重合
posteriori_dist = get_posteriori_dist(x)
plot_pdf(posteriori_dist, label="posteriori")
# verify_posteriori_dist(x, label="posteriori_bayes")

# 由于正态分布的特殊性，修正后的值刚好就是后验概率的顶点
theta = tweedie(x)
print(f"x={x.item():.4f} -> theta={theta.item():.4f}\n")
sns.scatterplot(x=theta.cpu(), y=0, marker="o", color="red")
plt.show()

# ===============================================================================


plot_target_score()
sns.scatterplot(x=x.cpu(), y=target_score(x).cpu(), marker="o", color="purple")
plt.show()
