# -*- coding: utf-8 -*-
# @file coefficient.py
# @author zhangshilong
# @date 2024/8/10

import math
from functools import cache

ALPHA = 0.95


@cache
def temp(t):
    return (1 - ALPHA) / (t + 1) + ALPHA


@cache
def alpha(t):
    return ALPHA
    # return temp(t) / temp(t - 1)


@cache
def beta(t):
    return 1 - alpha(t)


@cache
def alpha_prod(t):
    if t == 0:
        return 1
    else:
        return alpha(t) * alpha_prod(t - 1)


@cache
def sigma2(t):
    return (1 - alpha_prod(t - 1)) / (1 - alpha_prod(t)) * beta(t)


@cache
def sigma(t):
    return math.sqrt(sigma2(t))


@cache
def SNR(t):
    return alpha_prod(t) / (1 - alpha_prod(t))


# ==================================================================================================


@cache
def picture_weight_v1(t):
    return (alpha_prod(t - 1) * beta(t) ** 2) / (2 * sigma2(t) * (1 - alpha_prod(t)) ** 2)


@cache
def picture_weight_v2(t):
    return (SNR(t - 1) - SNR(t)) / 2


@cache
def noise_weight_v1(t):
    return beta(t) ** 2 / (2 * sigma2(t) * alpha(t) * (1 - alpha_prod(t)))


@cache
def noise_weight_v2(t):
    return picture_weight_v2(t) / SNR(t)


@cache
def score_weight_v1(t):
    return beta(t) ** 2 / (2 * sigma2(t) * alpha(t))


@cache
def score_weight_v2(t):
    return noise_weight_v2(t) * (1 - alpha_prod(t))


# ==================================================================================================


@cache
def picture_coefficient1(t):
    return math.sqrt(alpha(t)) * (1 - alpha_prod(t - 1)) / (1 - alpha_prod(t))


@cache
def picture_coefficient2(t):
    return math.sqrt(alpha_prod(t - 1)) * (1 - alpha(t)) / (1 - alpha_prod(t))


@cache
def noise_coefficient1(t):
    return 1 / (math.sqrt(alpha(t)))


@cache
def noise_coefficient2(t):
    return - (1 - alpha(t)) / (math.sqrt(alpha(t)) * math.sqrt(1 - alpha_prod(t)))


@cache
def score_coefficient1(t):
    return 1 / (math.sqrt(alpha(t)))


@cache
def score_coefficient2(t):
    return (1 - alpha(t)) / (math.sqrt(alpha(t)))


for t in range(2, 1000, 50):
    # print(
    #     f"{t=} {alpha(t)=:.4f} {alpha_prod(t)=:.4f} {SNR(t)=:.4f} {sigma2(t)=:.4f} || "
    #     f"{picture_weight_v1(t):.4f} {picture_weight_v2(t):.4f} || "
    #     f"{noise_weight_v1(t):.4f} {noise_weight_v2(t):.4f} {(1 - ALPHA) / (2 * ALPHA):.4f} || "
    #     f"{score_weight_v1(t):.4f} {score_weight_v2(t):.4f} {(1 - ALPHA) / (2 * ALPHA):.4f}"
    # )
    print(f"[{t=}] {sigma(t):.4f} || "
          f"{picture_coefficient1(t):.4f} {picture_coefficient2(t):.4f} || "
          f"{noise_coefficient1(t):.4f} {noise_coefficient2(t):.4f} || "
          f"{score_coefficient1(t):.4f} {score_coefficient2(t):.4f}")
