# -*- coding: utf-8 -*-
# @file classification_rule.py
# @author zhangshilong
# @date 2024/7/11

import re

import pandas as pd

from ...tools.constant import Category

companies = NotImplemented

pattern = "|".join(companies)


# 简单判断问题内有没有公司名，有就是Text，没有就是SQL
# 98/110(89.09%)，都是误把Text判断为SQL，10条使用公司简称，2条把"海看网络科技（山东）股份有限公司"写成"山东海看网络科技有限公司"
def func_v1(row):
    question = row["问题"]
    category = Category.TEXT if re.search(pattern, question) else Category.SQL
    return pd.Series({"问题分类": category})
