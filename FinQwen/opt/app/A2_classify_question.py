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
# 利用jaccard相似度给每个问题匹配最相似的公司
similarity_matrix = tokenizer.pairwise_jaccard_scores(questions, companies)  # shape: 1000*80
max_similar_scores = similarity_matrix.max(axis=1)
max_similar_indices = similarity_matrix.argmax(axis=1)
max_similar_companies = [companies[index] for index in max_similar_indices]
classification_df = question_df.copy()
classification_df["最大相似分数"] = max_similar_scores
classification_df["最相似公司名称"] = max_similar_companies

# 要求问题与公司的相似度大于0.1 或 相似度>0.05且不含基金、股票、A股、港股等关键字才认为是Text
cond1 = max_similar_scores > 0.1
cond2a = max_similar_scores > 0.05
cond2b = [re.search("基金|股票|A股|港股", question) is None for question in questions]
cond2 = np.logical_and(cond2a, cond2b)
is_text = np.logical_or(cond1, cond2)

classification_df["问题分类"] = np.where(is_text, Category.TEXT, Category.SQL)
classification_df["公司名称"] = np.where(is_text, max_similar_companies, np.nan)
# 上一步numpy将np.nan转为"nan"，修正它
classification_df.replace("nan", np.nan, inplace=True)
classification_df["公司id"] = classification_df["公司名称"].map(company_to_cid_mapping)

# 查看分类比例
category_counts = classification_df["问题分类"].value_counts()
print(f"{category_counts=}")  # 600个SQL, 400个Text

# 分数由低到高展示
classification_df.query(f"问题分类 == '{Category.TEXT}'").sort_values("最大相似分数")

# 分数由高到低展示
classification_df.query(f"问题分类 == '{Category.SQL}'").sort_values("最大相似分数", ascending=False)

# 保存
classification_df.drop(columns=["最大相似分数", "最相似公司名称"], inplace=True)
classification_df.to_csv(Config.QUESTION_CLASSIFICATION_PATH, index=False)
