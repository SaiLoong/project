# -*- coding: utf-8 -*-
# @file classify_test_question_by_rule.py
# @author zhangshilong
# @date 2024/7/7

import re

import numpy as np

from ..tools.config import Config
from ..tools.constant import Category

tokenizer = Config.get_tokenizer()

test_question_df = Config.get_test_question_df()
company_df, companies = Config.get_company_df(return_companies=True)

questions = test_question_df["问题"].tolist()

# 利用jaccard相似度匹配公司，阈值0.1
similarity_matrix = tokenizer.pairwise_scores(questions, companies)  # shape: 110*80
max_similar_scores = similarity_matrix.max(axis=1)
max_similar_indices = similarity_matrix.argmax(axis=1)
max_similar_companies = [companies[index] for index in max_similar_indices]


# 要求问题与公司的相似度大于阈值且不含基金、股票等关键字才认为是Text
threshold = 0.05
pattern = "基金|股票"
cond1 = max_similar_scores > threshold
cond2 = [re.search(pattern, question) is None for question in questions]
is_text = np.logical_and(cond1, cond2)

category_df = test_question_df.copy()
category_df["问题分类"] = np.where(is_text, Category.TEXT, Category.SQL)
category_df["公司名称识别"] = np.where(is_text, max_similar_companies, np.nan)
# 上一步numpy将np.nan转为"nan"，修正它
category_df.replace("nan", np.nan, inplace=True)

# =========================================================


# 展示分类效果
category_df["分类正确"] = category_df["问题标签"] == category_df["问题分类"]
question_num = len(category_df)
correct_num = category_df["分类正确"].sum()
print(f"测试问题数： {question_num}")  # 110
print(f"分类正确数：{correct_num}")
print(f"分类正确率：{correct_num / question_num:.2%}")
# 展示bad case
category_df.query("分类正确 == False")

# 展示公司名称识别效果
text_df = category_df.query(f"问题标签 == '{Category.TEXT}'").copy()
text_df["公司名称识别正确"] = text_df["公司名称"] == text_df["公司名称识别"]
text_num = len(text_df)
company_correct_num = text_df["公司名称识别正确"].sum()
print(f"测试Text问题数： {text_num}")
print(f"公司名称识别正确数：{company_correct_num}")
print(f"公司名称识别正确率：{company_correct_num / text_num:.2%}")
# 展示bad case
text_df.query("公司名称识别正确 == False")

"""
结论：threshold=0.05 + 基金、股票关键字过滤：110/110，54/54

threshold=0.1: 106/110，50/54          bad case: [53, 55, 94, 100]，4个用了公司简称的问题
threshold=0.05: 107/110，54/54         bad case: [25, 65, 75]，3个SQL问题与公司名撞字
"""
