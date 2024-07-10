# -*- coding: utf-8 -*-
# @file A2_classify_question.py
# @author zhangshilong
# @date 2024/7/7
# 利用规则对问题集分类。如果是Text，填上 公司名称 和 公司id

import re

import numpy as np

from ..tools.config import Config
from ..tools.constant import Category

tokenizer = Config.get_tokenizer()

question_df = Config.get_question_df()
questions = question_df["问题"].tolist()

company_df, companies = Config.get_company_df(return_companies=True)
company_to_cid_mapping = company_df.set_index(keys="公司名称")["公司id"].to_dict()

# 根据experiment.classify_test_question_by_rule.py的测试结果，最简单的规则效果反而最好
# 利用jaccard相似度对问题匹配公司，阈值0.1，外加一些关键词过滤
similarity_matrix = tokenizer.pairwise_scores(questions, companies)  # shape: 1000*80
max_similar_scores = similarity_matrix.max(axis=1)
max_similar_indices = similarity_matrix.argmax(axis=1)
max_similar_companies = [companies[index] for index in max_similar_indices]

# 要求问题与公司的相似度大于阈值且不含基金、股票等关键字才认为是Text
threshold = 0.05
pattern = "基金|股票"
cond1 = max_similar_scores > threshold
cond2 = [re.search(pattern, question) is None for question in questions]
is_text = np.logical_and(cond1, cond2)

category_df = question_df.copy()
category_df["问题分类"] = np.where(is_text, Category.TEXT, Category.SQL)
category_df["公司名称"] = np.where(is_text, max_similar_companies, np.nan)
# 上一步numpy将np.nan转为"nan"，修正它
category_df.replace("nan", np.nan, inplace=True)
category_df["公司id"] = category_df["公司名称"].map(company_to_cid_mapping)

# 查看分类比例
category_counts = category_df["问题分类"].value_counts()
print(f"{category_counts=}")  # 599个SQL, 401个Text

# 保存
category_df.to_csv(Config.QUESTION_CATEGORY_PATH, index=False)
