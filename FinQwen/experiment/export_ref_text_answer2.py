# -*- coding: utf-8 -*-
# @file export_ref_text_answer2.py
# @author zhangshilong
# @date 2024/7/20

import pandas as pd

from ..tools.config import Config

text_df = pd.read_json(f"{Config.EXPERIMENT_OUTPUT_DIR}/ref_text_answer_v5.json")

Config.export_submit_result(text_df, f"{Config.EXPERIMENT_OUTPUT_DIR}/ref_text_v5_submit_result.jsonl")
"""
v2    76.75
v3    76.42
v3.2  76.83（说明数字要加逗号）
v4    77.05
v5    77.16
"""

sql_df = Config.get_sql_answer_prediction_df()

mixed_df = pd.concat([text_df[["问题id", "答案"]], sql_df[["问题id", "答案"]]])
Config.export_submit_result(mixed_df, f"{Config.EXPERIMENT_OUTPUT_DIR}/mixed_v5_submit_result.jsonl")
