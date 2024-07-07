# -*- coding: utf-8 -*-
# @file A2_classify_question.py
# @author zhangshilong
# @date 2024/7/7
# 利用规则对问题集分类。如果是Text，填上 公司名称 和 公司id

import re

import pandas as pd

from ..tools.config import Config
from ..tools.constant import Category
from ..tools.constant import NA

question_df = Config.get_question_df()
company_df, companies = Config.get_company_df(return_companies=True)

# 根据experiment.classify_test_question_by_rule.py的测试结果，最简单的规则效果反而最好
# 简单判断问题内有没有公司名，有就是Text，没有就是SQL
pattern = "|".join(companies)
company_to_cid_mapping = company_df.set_index(keys="公司名称")["公司id"].to_dict()


def func(row):
    question = row["问题"]
    if m := re.search(pattern, question):
        category = Category.TEXT
        company = m.group()
        cid = company_to_cid_mapping[company]
    else:
        category = Category.SQL
        company = NA
        cid = NA

    return pd.Series({"分类": category, "公司名称": company, "公司id": cid})


category_df = pd.concat([question_df, question_df.progress_apply(func, axis=1)], axis=1)

# 查看分类比例
category_counts = category_df["分类"].value_counts()
print(f"{category_counts=}")  # 611个SQL, 389个Text

# 保存
category_df.to_csv(Config.QUESTION_CATEGORY_PATH, index=False)
