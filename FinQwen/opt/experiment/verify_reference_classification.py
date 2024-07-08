# -*- coding: utf-8 -*-
# @file verify_reference_classification.py
# @author zhangshilong
# @date 2024/7/8

import pandas as pd

from ..tools.config import Config

test_question_df = Config.get_test_question_df()

ref_question_category_df = pd.read_csv(f"{Config.EXPERIMENT_REFERENCE_DIR}/A01_question_classify.csv")
assert len(ref_question_category_df) == Config.QUESTION_NUM

# join
merge_df = pd.merge(test_question_df, ref_question_category_df[["问题id", "分类"]], on="问题id")

# 展示分类效果
merge_df["分类正确"] = merge_df["标签"] == merge_df["分类"]
print(f'分类正确数：{merge_df["分类正确"].sum()}')  # 98
merge_df.query("分类正确 == False")

"""
结论：正确率98%

正确分类是SQL，误判成Text：
    511: 2021年年度报告里，光大保德信基金管理有限公司管理的基金中，机构投资者持有份额比个人投资者多的基金有多少只?
    659: 2019年中期报告里，国融基金管理有限公司管理的基金中，机构投资者持有份额比个人投资者多的基金有多少只?
"""