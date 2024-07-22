# -*- coding: utf-8 -*-
# @file nl2sql_utils.py
# @author zhangshilong
# @date 2024/7/22

# TODO 先放这里

from constant import IGNORE_TOKEN_ID
from qwen_utils import DEFAULT_SYSTEM
from qwen_utils import make_context


# TODO 晚点再考虑要不要并入tokenizer方法
def build_finetune_data(tokenizer, question, answer, system=DEFAULT_SYSTEM):
    _, input_ids_a = make_context(tokenizer, question, system=system)
    dct_b = tokenizer(f"{answer}<|im_end|>\n")

    # 结尾没有加eod_token，但是以"<|im_end|>\n"结尾
    return dict(
        input_ids=input_ids_a + dct_b["input_ids"],
        attention_mask=[1] * len(input_ids_a) + dct_b["attention_mask"],
        labels=[IGNORE_TOKEN_ID] * len(input_ids_a) + dct_b["input_ids"]
    )
