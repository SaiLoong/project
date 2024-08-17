# -*- coding: utf-8 -*-
# @file ode.py
# @author zhangshilong
# @date 2024/8/14

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

beta = -2.
delta_t = 0.1


def func_x(t):
    return np.exp(-beta * t / 2)


# 解析解
t = np.arange(0, 10, 0.1)
x_true = func_x(t)
sns.lineplot(x=t, y=x_true)

# 离散过程
point_t = 0
point_x = func_x(point_t)
points_t = [point_t]
points_x = [point_x]

N = 20
for _ in range(N):
    point_t += delta_t
    point_x *= (1 - beta * delta_t / 2)
    points_t.append(point_t)
    points_x.append(point_x)

sns.scatterplot(x=points_t, y=points_x, marker="o", color="black")

plt.show()
