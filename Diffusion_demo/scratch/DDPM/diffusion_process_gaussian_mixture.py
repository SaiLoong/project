# -*- coding: utf-8 -*-
# @file diffusion_process_gaussian_mixture.py
# @author zhangshilong
# @date 2024/8/20

from functools import cache
from functools import partial

import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.distributions as D
from tqdm import trange

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
X_MIN = -20
X_MAX = 20
SAMPLE_SIZE = 100000


def convert(x):
    return x.to(DEVICE) if isinstance(x, torch.Tensor) \
        else torch.tensor(x, dtype=torch.float, device=DEVICE)


X0_PIS = convert([0.2, 0.2, 0.3, 0.1, 0.2])
X0_MUS = convert([-9, -5, 2, 6, 10])
X0_TAUS = convert([1, 0.7, 1.3, 0.8, 2.2])

# ===============================================================================


ALPHA = convert(0.98)

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


# ===============================================================================


def get_uniform_dist(low, high):
    low = convert(low)
    high = convert(high)

    return D.Uniform(low, high)


def get_gaussian_dist(mu, sigma):
    mu = convert(mu)
    sigma = convert(sigma)

    return D.Normal(mu, sigma)


def get_gaussian_mixture_dist(pis, mus, sigmas):
    pis = convert(pis)
    mus = convert(mus)
    sigmas = convert(sigmas)

    weights = D.Categorical(pis)
    dists = D.Normal(mus, sigmas)
    return D.MixtureSameFamily(weights, dists)


def get_x0_dist():
    return get_gaussian_mixture_dist(X0_PIS, X0_MUS, X0_TAUS)


def get_xt_from_xtm1_dist(xtm1, t):
    xtm1 = convert(xtm1)

    return get_gaussian_dist(torch.sqrt(alpha(t)) * xtm1, torch.sqrt(1 - alpha(t)))


def get_xt_from_x0_dist(x0, t):
    x0 = convert(x0)

    return get_gaussian_dist(torch.sqrt(bar_alpha(t)) * x0, torch.sqrt(1 - bar_alpha(t)))


def get_xt_dist(t):
    xt_pis = X0_PIS
    xt_mus = torch.sqrt(bar_alpha(t)) * X0_MUS
    xt_taus = torch.sqrt(bar_alpha(t) * X0_TAUS ** 2 + 1 - bar_alpha(t))
    return get_gaussian_mixture_dist(xt_pis, xt_mus, xt_taus)


def get_xtm1_from_xt_and_x0_dist(xt, x0, t):
    xt = convert(xt)
    x0 = convert(x0)

    mu = (torch.sqrt(alpha(t)) * (1 - bar_alpha(t - 1)) * xt +
          torch.sqrt(bar_alpha(t - 1)) * (1 - alpha(t)) * x0) \
         / (1 - bar_alpha(t))
    sigma = torch.sqrt((1 - alpha(t)) * (1 - bar_alpha(t - 1)) / (1 - bar_alpha(t)))
    return get_gaussian_dist(mu, sigma)


def get_x0_from_xt_prob(x0, xt, t):
    x0 = convert(x0)
    xt = convert(xt)
    assert not (x0.numel() > 1 and xt.numel() > 1), "只能其中一边为序列"

    x0_dist = get_x0_dist()
    xt_from_x0_dist = get_xt_from_x0_dist(x0, t)
    xt_dist = get_xt_dist(t)

    return (x0_dist.log_prob(x0) + xt_from_x0_dist.log_prob(xt) - xt_dist.log_prob(xt)).exp()


def get_xtm1_from_xt_prob(xtm1, xt, t):
    xtm1 = convert(xtm1)
    xt = convert(xt)
    assert not (xtm1.numel() > 1 and xt.numel() > 1), "只能其中一边为序列"

    xtm1_dist = get_xt_dist(t - 1)
    xt_from_xtm1_dist = get_xt_from_xtm1_dist(xtm1, t)
    xt_dist = get_xt_dist(t)

    return (xtm1_dist.log_prob(xtm1) + xt_from_xtm1_dist.log_prob(xt) - xt_dist.log_prob(xt)).exp()


def rejection_sampling(target_pdf, proposal_dist, factor):
    points = proposal_dist.sample([SAMPLE_SIZE])
    uniforms = get_uniform_dist(0, 1).sample([SAMPLE_SIZE])

    proposal_probs = proposal_dist.log_prob(points).exp()
    mask = factor * proposal_probs * uniforms < target_pdf(points)
    return points[mask]


# ===============================================================================

def plot_pdf(pdf, x_min=X_MIN, x_max=X_MAX, interval=0.01, factor=1, label=None):
    xs = torch.arange(x_min, x_max, interval, device=DEVICE)
    ys = pdf(xs) * factor
    sns.lineplot(x=xs.cpu(), y=ys.cpu(), label=label)


def plot_dist_pdf(dist, x_min=X_MIN, x_max=X_MAX, interval=0.01, factor=1, label=None):
    def pdf(x):
        return dist.log_prob(x).exp()

    plot_pdf(pdf, x_min, x_max, interval, factor, label)


def plot_rejection_sampling(target_pdf, proposal_dist, factor):
    plot_pdf(target_pdf, label="target")
    plot_dist_pdf(proposal_dist, factor=factor, label=f"proposal({factor=})")

    samples = rejection_sampling(target_pdf, proposal_dist, factor)
    plt.title(f"valid sample num: {len(samples)}")
    sns.histplot(samples.cpu(), stat="density", binwidth=0.1)
    plt.show()


# ===============================================================================


def verify_forward_process(T):
    x0_dist = get_x0_dist()
    plt.title(f"{T=}")
    plot_dist_pdf(x0_dist, label="x0")

    # 逐步迭代
    xts = x0_dist.sample([SAMPLE_SIZE])
    for t in trange(1, T + 1):
        xts = get_xt_from_xtm1_dist(xts, t).sample([])
    sns.histplot(xts.cpu(), stat="density", binwidth=0.1)

    # 一次迭代
    x0s = x0_dist.sample([SAMPLE_SIZE])
    xTs = get_xt_from_x0_dist(x0s, T).sample([])
    sns.histplot(xTs.cpu(), stat="density", binwidth=0.1)

    xt_dist = get_xt_dist(T)
    plot_dist_pdf(xt_dist, label=f"x{T}")

    standard_gaussian_dist = get_gaussian_dist(0, 1)
    plot_dist_pdf(standard_gaussian_dist, label="standard_gaussian")

    plt.show()


# proposal_dist和factor只能靠试了
def verify_rejection_sampling(xt, t, mu, sigma, factor):
    xt = convert(xt)
    target_pdf = partial(get_xtm1_from_xt_prob, xt=xt, t=t)
    proposal_dist = get_gaussian_dist(mu, sigma)
    plot_rejection_sampling(target_pdf, proposal_dist, factor)


# verify_rejection_sampling(xt=0, t=200, mu=-1.5, sigma=8, factor=3)


def verify_x0_from_xt(x0, t):
    x0_dist = get_x0_dist()
    xt_dist = get_xt_dist(t)

    x0 = convert(x0)
    value1 = x0_dist.log_prob(x0).exp()

    xts = xt_dist.sample([SAMPLE_SIZE])
    value2 = get_x0_from_xt_prob(x0, xts, t).mean()

    print(f"{value1=} {value2=}")


def verify_xtm1_from_xt(xtm1, t):
    xtm1_dist = get_xt_dist(t - 1)
    xt_dist = get_xt_dist(t)

    xtm1 = convert(xtm1)
    value1 = xtm1_dist.log_prob(xtm1).exp()

    xts = xt_dist.sample([SAMPLE_SIZE])
    value2 = get_xtm1_from_xt_prob(xtm1, xts, t).mean()

    print(f"{value1=} {value2=}")


# 展示x0和xt之间的四个分布
def between_x0_and_xt(t, x0, xt):
    x0_dist = get_x0_dist()
    plot_dist_pdf(x0_dist, label="x0")

    xt_dist = get_xt_dist(t)
    plot_dist_pdf(xt_dist, label=f"x{t}")

    xt_from_x0_dist = get_xt_from_x0_dist(x0=x0, t=t)
    plot_dist_pdf(xt_from_x0_dist, label=f"x{t}_from_x0")

    x0_from_xt_pdf = partial(get_x0_from_xt_prob, xt=xt, t=t)
    plot_pdf(x0_from_xt_pdf, label=f"x0_from_x{t}")

    standard_gaussian_dist = get_gaussian_dist(0, 1)
    plot_dist_pdf(standard_gaussian_dist, label="standard_gaussian")

    plt.show()


# 展示xtm1和xt之间的四个分布，外加q(xtm1|xt,x0)
def between_xtm1_and_xt(t, xtm1, xt, x0=None, x_min=X_MIN, x_max=X_MAX):
    xtm1_dist = get_xt_dist(t - 1)
    plot_dist_pdf(xtm1_dist, x_min, x_max, label=f"x{t - 1}")

    xt_dist = get_xt_dist(t)
    plot_dist_pdf(xt_dist, x_min, x_max, label=f"x{t}")

    xt_from_xtm1_dist = get_xt_from_xtm1_dist(xtm1=xtm1, t=t)
    plot_dist_pdf(xt_from_xtm1_dist, x_min, x_max, label=f"x{t}_from_x{t - 1}")

    xtm1_from_xt_pdf = partial(get_xtm1_from_xt_prob, xt=xt, t=t)
    plot_pdf(xtm1_from_xt_pdf, x_min, x_max, label=f"x{t - 1}_from_x{t}")

    if x0 is not None:
        xtm1_from_xt_and_x0_dist = get_xtm1_from_xt_and_x0_dist(xt, x0, t)
        plot_dist_pdf(xtm1_from_xt_and_x0_dist, x_min, x_max, label=f"x{t - 1}_from_x{t}_and_x0")

    standard_gaussian_dist = get_gaussian_dist(0, 1)
    plot_dist_pdf(standard_gaussian_dist, x_min, x_max, label="standard_gaussian")

    plt.show()


# 展示p(xt)逐渐变化的过程
def show_xt_evolution(T, step=1):
    x0_dist = get_x0_dist()
    plot_dist_pdf(x0_dist, label="x0")

    for t in range(1, T + 1, step):
        xt_dist = get_xt_dist(t)
        plot_dist_pdf(xt_dist, label=f"x{t}")

    standard_gaussian_dist = get_gaussian_dist(0, 1)
    plot_dist_pdf(standard_gaussian_dist, label="standard_gaussian")

    plt.show()


# 展示q(x0|xt)逐渐变化的过程
def show_x0_from_xt_evolution(T, xt, step=1, x_min=X_MIN, x_max=X_MAX):
    for t in range(1, T + 1, step):
        x0_from_xt_pdf = partial(get_x0_from_xt_prob, xt=xt, t=t)
        plot_pdf(x0_from_xt_pdf, x_min, x_max, label=f"x0_from_x{t}")

    x0_dist = get_x0_dist()
    plot_dist_pdf(x0_dist, x_min, x_max, label="x0")

    plt.show()


# ===============================================================================


# verify_forward_process(T=50)
# verify_x0_from_xt(x0=2, t=50)
# verify_xtm1_from_xt(xtm1=2, t=50)

# between_x0_and_xt(t=200, x0=2, xt=4)
between_xtm1_and_xt(t=10, xtm1=2, xt=4, x0=0, x_min=-5, x_max=5)

# show_xt_evolution(T=101, step=20)
# show_x0_from_xt_evolution(T=200, xt=-5, step=50, x_min=-15, x_max=15)
