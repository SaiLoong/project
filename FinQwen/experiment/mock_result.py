# -*- coding: utf-8 -*-
# @file mock_result.py
# @author zhangshilong
# @date 2024/7/8

from ..tools.config import Config
from ..tools.utils import File

question_df = Config.get_question_df(rename=False)

empty_question_df = question_df.copy()
empty_question_df["answer"] = ""
File.dataframe_to_jsonl(empty_question_df, f"{Config.EXPERIMENT_OUTPUT_DIR}/empty_submit_result.jsonl")

repeat_question_df = question_df.copy()
repeat_question_df["answer"] = question_df["question"]
File.dataframe_to_jsonl(repeat_question_df, f"{Config.EXPERIMENT_OUTPUT_DIR}/repeat_submit_result.jsonl")

"""
总结：
直接将答案填空字符串，0分
直接把问题填进答案，29.88分（35.85/20.92）
"""
