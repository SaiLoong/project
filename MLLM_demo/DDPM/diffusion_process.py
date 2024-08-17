# -*- coding: utf-8 -*-
# @file diffusion_process.py
# @author zhangshilong
# @date 2024/8/8

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

DATA_SIZE = 10000
ALPHA = 0.97


# 初始分布
def get_initial_data_v1():
    pi = np.random.binomial(n=1, p=0.3, size=DATA_SIZE)
    data = pi * np.random.normal(loc=-2, scale=0.2, size=DATA_SIZE) + \
           (1 - pi) * np.random.normal(loc=2, scale=1, size=DATA_SIZE)
    return data


def get_initial_data_v2():
    data = np.random.uniform(size=DATA_SIZE) + np.random.standard_exponential(size=DATA_SIZE)
    return data


# 迭代一轮
def iter_once(data):
    return np.sqrt(ALPHA) * data + \
        np.sqrt(1 - ALPHA) * np.random.normal(loc=0, scale=1, size=DATA_SIZE)


# 迭代多轮
def iter_n_v1(data, n):
    for _ in range(n):
        data = iter_once(data)
    return data


# 迭代多轮优化
def iter_n_v2(data, n):
    alpha = ALPHA ** n
    return np.sqrt(alpha) * data + \
        np.sqrt(1 - alpha) * np.random.normal(loc=0, scale=1, size=DATA_SIZE)


ITER_NUM = 50
normal_data = np.random.normal(loc=0, scale=1, size=DATA_SIZE)
init_data = get_initial_data_v2()
final_data_v1 = iter_n_v1(init_data, n=ITER_NUM)
final_data_v2 = iter_n_v2(init_data, n=ITER_NUM)

sns.kdeplot([normal_data, init_data, final_data_v1, final_data_v2])
plt.show()
# 无论初始分布是啥，足够多轮迭代后都接近标准正态分布
