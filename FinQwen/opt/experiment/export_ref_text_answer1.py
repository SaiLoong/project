# -*- coding: utf-8 -*-
# @file export_ref_text_answer1.py
# @author zhangshilong
# @date 2024/7/20

import pandas as pd

from ..tools.config import Config
from ..tools.constant import Category
from ..tools.utils import File

ref_df = pd.read_csv(f"{Config.REFERENCE_DIR}/FA_V5_Text_cap4_4_nt.csv")
ref_df = ref_df[["问题id", "实体答案", "final_ans1"]]
ref_df.rename(columns={"实体答案": "ref_公司名称", "final_ans1": "答案"}, inplace=True)

classification_df = Config.get_question_classification_df(category=Category.TEXT)
classification_df.drop(columns="公司id", inplace=True)
text_df = pd.merge(classification_df, ref_df, how="left", on="问题id")
text_df.sort_values(["ref_公司名称", "问题id"], inplace=True)
text_df.reset_index(drop=True, inplace=True)

File.dataframe_to_json(text_df, f"{Config.EXPERIMENT_OUTPUT_DIR}/ref_text_answer.json")

Config.export_submit_result(text_df, f"{Config.EXPERIMENT_OUTPUT_DIR}/ref_text_submit_result.jsonl")  # 76.37
