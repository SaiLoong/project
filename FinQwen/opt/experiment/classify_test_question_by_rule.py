# -*- coding: utf-8 -*-
# @file classify_test_question_by_rule.py
# @author zhangshilong
# @date 2024/7/7

import re

import pandas as pd

from ..tools.config import Config
from ..tools.constant import Category

test_question_df = Config.get_test_question_df()
company_df, companies = Config.get_company_df(return_companies=True)

# 简单判断问题内有没有公司名，有就是Text，没有就是SQL
pattern = "|".join(companies)


def func(row):
    question = row["问题"]
    category = Category.TEXT if re.search(pattern, question) else Category.SQL
    return pd.Series({"问题分类": category})


category_df = pd.concat([test_question_df, test_question_df.progress_apply(func, axis=1)], axis=1)

# 展示分类效果
category_df["分类正确"] = category_df["问题标签"] == category_df["问题分类"]
question_num = len(category_df)
correct_num = category_df["分类正确"].sum()
print(f"测试问题数： {question_num}")  # 110
print(f"分类正确数：{correct_num}")  # 98
print(f"分类正确率：{correct_num / question_num:.2%}")  # 89.09%
# 展示bad case
category_df.query("分类正确 == False")

"""
结论：分类正确率 89.09%, 98/110

都是误把Text判断为SQL，10条使用公司简称，2条把"海看网络科技（山东）股份有限公司"写成"山东海看网络科技有限公司"
"""
