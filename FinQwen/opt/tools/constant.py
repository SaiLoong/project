# -*- coding: utf-8 -*-
# @file constant.py
# @author zhangshilong
# @date 2024/7/7

class Enum:
    @classmethod
    def items(cls):
        return {(k, v) for k, v in vars(cls).items()
                if not k.startswith("_") and not isinstance(v, (classmethod, list, set))}

    @classmethod
    def keys(cls):
        return {k for k, v in vars(cls).items()
                if not k.startswith("_") and not isinstance(v, (classmethod, list, set))}

    @classmethod
    def values(cls):
        return {v for k, v in vars(cls).items()
                if not k.startswith("_") and not isinstance(v, (classmethod, list, set))}


class Label(Enum):
    TEXT = "Text"
    SQL = "SQL"


Category = Label


class ModelMode(Enum):
    TRAIN = "train"
    EVAL = "eval"
