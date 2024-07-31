# -*- coding: utf-8 -*-
# @file function.py
# @author zhangshilong
# @date 2024/7/7
# 写了但是最后没用上的函数

import jsonlines


def read_jsonl(path):
    with jsonlines.open(path, "r") as f:
        return list(f.iter(type=dict, skip_invalid=True))


def write_jsonl(path, data):
    with jsonlines.open(path, "w") as f:
        f.write_all(data)
