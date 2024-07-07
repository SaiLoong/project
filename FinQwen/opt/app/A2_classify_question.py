# -*- coding: utf-8 -*-
# @file A2_classify_question.py
# @author zhangshilong
# @date 2024/7/7
# 利用规则对问题集分类

import re

import pandas as pd

from ..tools.config import Config
from ..tools.constant import Category

question_df = Config.get_question_df()
company_df, companies = Config.get_company_df(return_companies=True)

# 根据experiment.classify_test_question_by_rule.py的测试结果，最简单的规则效果反而最好
# 简单判断问题内有没有公司名，有就是Text，没有就是SQL
pattern = "|".join(companies)


def func(row):
    question = row["问题"]
    category = Category.TEXT if re.search(pattern, question) else Category.SQL
    return pd.Series({"分类": category})


category_df = pd.concat([question_df, question_df.progress_apply(func, axis=1)], axis=1)

# 保存
category_df.to_csv(Config.QUESTION_CATEGORY_PATH, index=False)
