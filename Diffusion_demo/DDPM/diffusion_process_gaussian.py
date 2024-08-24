# -*- coding: utf-8 -*-
# @file diffusion_process_gaussian.py
# @author zhangshilong
# @date 2024/8/24

from functools import cache
from functools import partial

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.distributions as D
from tqdm import trange

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
X_MIN = -5
X_MAX = 5
SAMPLE_SIZE = 100000


def convert(x):
    return x.to(DEVICE) if isinstance(x, torch.Tensor) \
        else torch.tensor(x, dtype=torch.float, device=DEVICE)


X0_MU = convert(-3)
X0_TAU = convert(1.5)

# ===============================================================================

# ALPHA = convert(0.98)

T_ = 1000
ALPHA_1 = convert(1 - 1e-4)
ALPHA_T = convert(1 - 0.02)


@cache
def alpha(t):
    # return ALPHA
    return (ALPHA_T - ALPHA_1) / (T_ - 1) * (t - 1) + ALPHA_1


@cache
def bar_alpha(t):
    return convert(1) if t == 0 else alpha(t) * bar_alpha(t - 1)


@cache
def xt_from_xtm1_factor(t):
    return torch.sqrt(alpha(t))


@cache
def xt_from_xtm1_std(t):
    return torch.sqrt(1 - alpha(t))


@cache
def xt_from_x0_factor(t):
    return torch.sqrt(bar_alpha(t))


@cache
def xt_from_x0_std(t):
    return torch.sqrt(1 - bar_alpha(t))


@cache
def xt_mean(t):
    return xt_from_x0_factor(t) * X0_MU


@cache
def xt_std(t):
    return torch.sqrt(
        (xt_from_x0_factor(t) * X0_TAU) ** 2 +
        xt_from_x0_std(t) ** 2)


# ===============================================================================


def get_uniform_dist(low, high):
    low = convert(low)
    high = convert(high)

    return D.Uniform(low, high)


def get_gaussian_dist(mu, sigma):
    mu = convert(mu)
    sigma = convert(sigma)

    return D.Normal(mu, sigma)


def get_standard_gaussian_dist():
    return get_gaussian_dist(0, 1)


def _calculate_gaussian_conjugate_mean_and_std(x, mu, tau, sigma, k=1.):
    # 每个x独立决定一个后验分布
    x = convert(x)
    mu = convert(mu)
    tau = convert(tau)
    sigma = convert(sigma)
    k = convert(k)

    # 先假设一个新的随机变量eta=k*theta，这样均值和标准差都是theta的k倍（正态分布的特性）
    # 套用共轭先验分布的公式得出p(eta|x)的分布后，将均值和标准差除以k还原得到p(theta|x)
    tau = k * tau
    mu = k * mu

    sigma0_2 = sigma ** 2

    sigma0_m2 = 1 / sigma0_2
    tau_m2 = 1 / tau ** 2

    sigma1_2 = 1 / (sigma0_m2 + tau_m2)
    mu1 = sigma1_2 * (x * sigma0_m2 + mu * tau_m2)
    sigma1 = torch.sqrt(sigma1_2)
    return mu1 / k, sigma1 / k


def get_gaussian_conjugate_dist(x, mu, tau, sigma, k=1.):
    mean, std = _calculate_gaussian_conjugate_mean_and_std(x, mu, tau, sigma, k)
    return get_gaussian_dist(mean, std)


def get_x0_dist():
    return get_gaussian_dist(X0_MU, X0_TAU)


def get_xt_from_xtm1_dist(xtm1, t):
    xtm1 = convert(xtm1)

    return get_gaussian_dist(xt_from_xtm1_factor(t) * xtm1, xt_from_xtm1_std(t))


def get_xt_from_x0_dist(x0, t):
    x0 = convert(x0)

    return get_gaussian_dist(xt_from_x0_factor(t) * x0, xt_from_x0_std(t))


def get_xt_dist(t):
    return get_gaussian_dist(xt_mean(t), xt_std(t))


def get_x0_from_xt_dist(xt, t):
    return get_gaussian_conjugate_dist(
        xt, mu=X0_MU, tau=X0_TAU,
        sigma=xt_from_x0_std(t), k=xt_from_x0_factor(t)
    )


def get_x0_from_xt_pdf_bayes(x0, xt, t):
    x0 = convert(x0)
    xt = convert(xt)
    assert not (x0.numel() > 1 and xt.numel() > 1), "只能其中一边为序列"

    x0_dist = get_x0_dist()
    xt_from_x0_dist = get_xt_from_x0_dist(x0, t)
    xt_dist = get_xt_dist(t)

    return (x0_dist.log_prob(x0) + xt_from_x0_dist.log_prob(xt) - xt_dist.log_prob(xt)).exp()


def get_xtm1_from_xt_dist(xt, t):
    return get_gaussian_conjugate_dist(
        xt, xt_mean(t - 1), xt_std(t - 1),
        sigma=xt_from_xtm1_std(t), k=xt_from_xtm1_factor(t)
    )


def get_xtm1_from_xt_pdf_bayes(xtm1, xt, t):
    xtm1 = convert(xtm1)
    xt = convert(xt)
    assert not (xtm1.numel() > 1 and xt.numel() > 1), "只能其中一边为序列"

    xtm1_dist = get_xt_dist(t - 1)
    xt_from_xtm1_dist = get_xt_from_xtm1_dist(xtm1, t)
    xt_dist = get_xt_dist(t)

    return (xtm1_dist.log_prob(xtm1) + xt_from_xtm1_dist.log_prob(xt) - xt_dist.log_prob(xt)).exp()


# TODO test
def get_xtm1_from_xt_and_x0_dist(xt, x0, t):
    xt = convert(xt)
    x0 = convert(x0)

    mu = (torch.sqrt(alpha(t)) * (1 - bar_alpha(t - 1)) * xt +
          torch.sqrt(bar_alpha(t - 1)) * (1 - alpha(t)) * x0) \
         / (1 - bar_alpha(t))
    sigma = torch.sqrt((1 - alpha(t)) * (1 - bar_alpha(t - 1)) / (1 - bar_alpha(t)))
    return get_gaussian_dist(mu, sigma)


# ===============================================================================


def plot_pdf(pdf, x_min=X_MIN, x_max=X_MAX, interval=0.01, factor=1, label=None):
    xs = torch.arange(x_min, x_max, interval, device=DEVICE)
    ys = pdf(xs) * factor
    sns.lineplot(x=xs.cpu(), y=ys.cpu(), label=label)


def plot_dist_pdf(dist, x_min=X_MIN, x_max=X_MAX, interval=0.01, factor=1, label=None):
    def pdf(x):
        return dist.log_prob(x).exp()

    plot_pdf(pdf, x_min, x_max, interval, factor, label)


# ===============================================================================

def verify_x0_dist():
    x0_dist = get_x0_dist()
    plot_dist_pdf(x0_dist, label="x0")

    xt_dist = get_xt_dist(0)
    plot_dist_pdf(xt_dist, label="xt (t=0)")

    plt.show()


def verify_x0_from_xt_dist(xt, t):
    x0_from_xt_dist = get_x0_from_xt_dist(xt, t)
    plot_dist_pdf(x0_from_xt_dist, label=f"x0_from_x{t}")

    x0_from_xt_pdf_bayes = partial(get_x0_from_xt_pdf_bayes, xt=xt, t=t)
    plot_pdf(x0_from_xt_pdf_bayes, label=f"x0_from_x{t} bayes")

    plt.show()


def verify_xtm1_from_xt_dist(xt, t):
    xtm1_from_xt_dist = get_xtm1_from_xt_dist(xt, t)
    plot_dist_pdf(xtm1_from_xt_dist, label=f"x{t - 1}_from_x{t}")

    xtm1_from_xt_pdf_bayes = partial(get_xtm1_from_xt_pdf_bayes, xt=xt, t=t)
    plot_pdf(xtm1_from_xt_pdf_bayes, label=f"x{t - 1}_from_x{t} bayes")

    plt.show()


# verify_x0_dist()
# verify_x0_from_xt_dist(xt=8, t=1000)
# verify_xtm1_from_xt_dist(xt=8, t=200)


def x0_to_xt_evolution(t):
    x0_dist = get_x0_dist()
    plot_dist_pdf(x0_dist, label="x0")

    # 走一大步
    x0s = x0_dist.sample([SAMPLE_SIZE])
    xts = get_xt_from_x0_dist(x0s, t).sample([])
    sns.histplot(xts.cpu(), stat="density", binwidth=0.1)

    # 走t小步
    xs = x0_dist.sample([SAMPLE_SIZE])
    for s in trange(1, t + 1):
        xs = get_xt_from_xtm1_dist(xs, s).sample([])
    sns.histplot(xs.cpu(), stat="density", binwidth=0.1)

    # ground truth
    xt_dist = get_xt_dist(t)
    plot_dist_pdf(xt_dist, label=f"x{t}")

    plt.show()


def x0_to_xt_to_x0_evolution(t):
    # ground truth
    x0_dist = get_x0_dist()
    plot_dist_pdf(x0_dist, label="x0")
    xt_dist = get_xt_dist(t)
    plot_dist_pdf(xt_dist, label=f"x{t}")

    # 走大步转一圈
    x0s = x0_dist.sample([SAMPLE_SIZE])
    xts = get_xt_from_x0_dist(x0s, t).sample([])
    sns.histplot(xts.cpu(), stat="density", binwidth=0.1)
    new_x0s = get_x0_from_xt_dist(xts, t).sample([])
    sns.histplot(new_x0s.cpu(), stat="density", binwidth=0.1)

    # 走小步转一圈
    xs = x0_dist.sample([SAMPLE_SIZE])
    for s in trange(1, t + 1):
        xs = get_xt_from_xtm1_dist(xs, s).sample([])
    sns.histplot(xs.cpu(), stat="density", binwidth=0.1)
    for s in trange(t, 0, -1):
        xs = get_xtm1_from_xt_dist(xs, s).sample([])
    sns.histplot(xs.cpu(), stat="density", binwidth=0.1)

    plt.show()


# 如果只取均值，最后样本会往p(x0)的中心靠拢，t越大越明显
def xt_to_x0_evolution(t, random=True):
    xt_dist = get_xt_dist(t)

    # 小步走
    xs = xt_dist.sample([SAMPLE_SIZE])
    for s in trange(t, 0, -1):
        xtm1_from_xt_dist = get_xtm1_from_xt_dist(xs, s)
        if random:
            xs = xtm1_from_xt_dist.sample([])
        else:
            xs = xtm1_from_xt_dist.mean

        if s % (t // 5) == 0 or s == 1:
            plot_dist_pdf(xt_dist, label=f"x{t}")

            sns.histplot(xs.cpu(), stat="density", binwidth=0.1)
            xsm1_dist = get_xt_dist(s - 1)
            plot_dist_pdf(xsm1_dist, label=f"x{s - 1}")
            plt.show()


# x0_to_xt_evolution(t=500)
# x0_to_xt_to_x0_evolution(t=600)
# xt_to_x0_evolution(t=500, random=True)


def x0_to_xt_evolution_fix_x0(t, x0):
    x0_dist = get_x0_dist()
    plot_dist_pdf(x0_dist, label="x0")

    # 走一大步
    x0s = convert(x0).repeat(SAMPLE_SIZE)
    xts = get_xt_from_x0_dist(x0s, t).sample([])
    sns.histplot(xts.cpu(), stat="density", binwidth=0.1)

    # 走t小步
    xs = convert(x0).repeat(SAMPLE_SIZE)
    for s in trange(1, t + 1):
        xs = get_xt_from_xtm1_dist(xs, s).sample([])
    sns.histplot(xs.cpu(), stat="density", binwidth=0.1)

    # ground truth
    xt_from_x0_dist = get_xt_from_x0_dist(x0, t)
    plot_dist_pdf(xt_from_x0_dist, label=f"x{t}_from_x0")

    standard_gaussian_dist = get_standard_gaussian_dist()
    plot_dist_pdf(standard_gaussian_dist, label="standard_gaussian")

    plt.show()


# 结果类似，样本会往p(xt|x0)的中心靠拢，而这个中心又在x0附近
def xt_to_x0_evolution_fix_x0(t, x0, random=True):
    x0_dist = get_x0_dist()
    plot_dist_pdf(x0_dist, label="x0")
    xt_from_x0_dist = get_xt_from_x0_dist(x0, t)
    plot_dist_pdf(xt_from_x0_dist, label=f"x{t}_from_x0")
    standard_gaussian_dist = get_standard_gaussian_dist()
    plot_dist_pdf(standard_gaussian_dist, label="standard_gaussian")

    # 初始化xt，基于固定的x_0
    xs = xt_from_x0_dist.sample([SAMPLE_SIZE])
    sns.histplot(xs.cpu(), stat="density", binwidth=0.1)
    plt.show()

    # 从t到2，最后得到的是p(x1|x0)
    for s in trange(t, 1, -1):
        xtm1_from_xt_and_x0_dist = get_xtm1_from_xt_and_x0_dist(xs, x0, s)
        if random:
            xs = xtm1_from_xt_and_x0_dist.sample([])
        else:
            xs = xtm1_from_xt_and_x0_dist.mean

        if s % (t // 5) == 0 or s == 2:
            plot_dist_pdf(xt_from_x0_dist, label=f"x{t}_from_x0")

            sns.histplot(xs.cpu(), stat="density", binwidth=0.1)
            xsm1_from_x0_dist = get_xt_from_x0_dist(x0, s - 1)
            plot_dist_pdf(xsm1_from_x0_dist, label=f"x{s - 1}_from_x0")
            plt.show()


# x0_to_xt_evolution_fix_x0(t=800, x0=3)
xt_to_x0_evolution_fix_x0(t=800, x0=3, random=True)
