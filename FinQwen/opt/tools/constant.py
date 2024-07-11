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


class ModelName(Enum):
    QWEN_1_8B_CHAT = "Qwen-1_8B-Chat"
    QWEN_7B_CHAT = "Qwen-7B-Chat"
    QWEN_14B_CHAT_INT4 = "Qwen-14B-Chat-Int4"
    QWEN_14B_CHAT_INT8 = "Qwen-14B-Chat-Int8"

    # TONGYI_FINANCE_14B_CHAT = "Tongyi-Finance-14B-Chat"
    TONGYI_FINANCE_14B_CHAT_INT4 = "Tongyi-Finance-14B-Chat-Int4"


class Category(Enum):
    TEXT = "Text"
    SQL = "SQL"


class ModelMode(Enum):
    TRAIN = "train"
    EVAL = "eval"
