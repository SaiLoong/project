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
    return pd.Series({"分类": category})


category_df = pd.concat([test_question_df, test_question_df.progress_apply(func, axis=1)], axis=1)

# 展示分类效果
category_df["分类正确"] = category_df["标签"] == category_df["分类"]
print(f'分类正确数：{category_df["分类正确"].sum()}')
category_df.query("分类正确 == False")

"""
结论：分类正确率 100%
"""