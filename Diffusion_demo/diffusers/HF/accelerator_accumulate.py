# -*- coding: utf-8 -*-
# @file accelerator_accumulate.py
# @author zhangshilong
# @date 2024/9/3

import torch
from torch import nn
from torch import optim
from torch.utils import data

from accelerate import Accelerator

# 参数
LABEL_NUM = 5
DATA_SIZE = 12
BATCH_SIZE = 1
EPOCH_NUM = 3


class Lambda(nn.Module):
    def __init__(self, func, *args, **kwargs):
        super().__init__()
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def forward(self, X):
        return self.func(X, *self.args, **self.kwargs)


def get_dataloader():
    Xs = torch.randint(LABEL_NUM, size=(DATA_SIZE, 1))
    Ys = 31.5 * torch.sin(Xs * 56.8) + torch.normal(mean=3.7, std=1.414, size=Xs.shape)
    # 记得保证Xs、Ys、预测值都是(n,1)形状，不然计算loss时会发生广播
    assert Xs.shape == Ys.shape
    dataset = data.TensorDataset(Xs, Ys)
    dataloader = data.DataLoader(dataset, BATCH_SIZE, shuffle=True)
    return dataloader


def get_model():
    model = nn.Sequential(
        nn.Embedding(LABEL_NUM, embedding_dim=32),
        Lambda(torch.squeeze, 1),
        nn.LazyLinear(out_features=1)
    )
    return model


def get_scheduler(_optimizer):
    def lmd(last_epoch: int):
        return (EPOCH_NUM - last_epoch) / EPOCH_NUM

    scheduler = optim.lr_scheduler.LambdaLR(_optimizer, lmd)
    return scheduler


dataloader = get_dataloader()
model = get_model()
optimizer = optim.AdamW(model.parameters(), lr=0.1)
scheduler = get_scheduler(optimizer)
print(f"[before] {type(dataloader)=} {type(model)=} {type(optimizer)=} {type(scheduler)=}")

accelerator = Accelerator(gradient_accumulation_steps=4)
dataloader, model, optimizer, scheduler = accelerator.prepare(dataloader, model, optimizer, scheduler)
print(f"[after] {type(dataloader)=} {type(model)=} {type(optimizer)=} {type(scheduler)=}")
"""
dataloader: accelerate.data_loader.DataLoaderShard
model: torch.nn.modules.container.Sequential (只有它没变)
optimizer: accelerate.optimizer.AcceleratedOptimizer
scheduler: accelerate.scheduler.AcceleratedScheduler
"""

w = model[0].weight
loss_func = nn.MSELoss()

for idx, (x, y) in enumerate(dataloader):
    # context通过控制accelerator.sync_gradients属性来控制optimizer是否更新参数和清理梯度
    # 即每4个substep中，前3个sync_gradients=False，只有最后一个是True
    # 特别地，最后一个substep一定sync_gradients=True
    # TODO 个人测试不加model好像没啥不同，可能在分布式训练才会有区别，先跟风写上
    with accelerator.accumulate(model):
        y_pred = model(x)
        loss = loss_func(y_pred, y)

        # 不受sync_gradients影响，每个substep都回传梯度，但是会除以gradient_accumulation_steps
        accelerator.backward(loss)

        # 下面三个方法都仅在sync_gradients=True时生效
        optimizer.step()
        scheduler.step()  # pytorch规定，scheduler.step()必须在optimizer.step()后面
        optimizer.zero_grad()
