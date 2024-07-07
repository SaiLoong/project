# -*- coding: utf-8 -*-
# @file mock_result.py
# @author zhangshilong
# @date 2024/7/8

from ..tools.config import Config

question_df = Config.get_question_df(rename=False)

question_df["answer"] = question_df["question"]

question_df.to_json(f"{Config.EXPERIMENT_OUTPUT_DIR}/mock_submit_result.jsonl", orient="records", force_ascii=False,
                    lines=True)

"""
总结：
直接将答案填空字符串，0分
直接把问题填进答案，29.88分（35.85/20.92）
"""