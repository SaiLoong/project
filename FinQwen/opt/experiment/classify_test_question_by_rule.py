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

# 利用jaccard相似度给每个问题匹配最相似的公司
similarity_matrix = tokenizer.pairwise_jaccard_scores(questions, companies)  # shape: 115*80
max_similar_scores = similarity_matrix.max(axis=1)
max_similar_indices = similarity_matrix.argmax(axis=1)
max_similar_companies = [companies[index] for index in max_similar_indices]
category_df = test_question_df.copy()
category_df["最大相似分数"] = max_similar_scores
category_df["最相似公司名称"] = max_similar_companies

# 分数由低到高展示
category_df.query(f"问题标签 == '{Category.TEXT}'").sort_values("最大相似分数")
# 分数由高到低展示
category_df.query(f"问题标签 == '{Category.SQL}'").sort_values("最大相似分数", ascending=False)

# 要求问题与公司的相似度大于0.1 或 相似度>0.05且不含基金、股票、A股、港股等关键字才认为是Text
cond1 = max_similar_scores > 0.1
cond2a = max_similar_scores > 0.05
cond2b = [re.search("基金|股票|A股|港股", question) is None for question in questions]
cond2 = np.logical_and(cond2a, cond2b)
is_text = np.logical_or(cond1, cond2)

category_df["问题分类"] = np.where(is_text, Category.TEXT, Category.SQL)
category_df["公司名称识别"] = np.where(is_text, max_similar_companies, np.nan)
# 上一步numpy将np.nan转为"nan"，修正它
category_df.replace("nan", np.nan, inplace=True)

# =========================================================


# 展示分类效果
category_df["分类正确"] = category_df["问题标签"] == category_df["问题分类"]
question_num = len(category_df)
correct_num = category_df["分类正确"].sum()
print(f"测试问题数： {question_num}")  # 115
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
结论：相似度>0.1 or (相似度>0.05 and 不含基金、股票、A股、港股)：115/115，55/55

相似度>0.1: 106/110，50/54          bad case: [53, 55, 94, 100]，4个用了公司简称的问题，相似度在0.06左右
相似度>0.05: 107/110，54/54         bad case: [25, 65, 75]，3个SQL问题与公司名撞字
相似度>0.05 and 不含基金、股票：110/111，54/55   bad case: [80]，'青海互助青稞酒股份有限公司本次发行股票的数量、每股面值分别是多少？'，本身相似度0.4

因为有几条真Text的问题相似度在0.06左右（用了公司简称），如果单纯地把阈值降至0.05，有几十个真SQL相似度稍微高于0.05的问题会被错判为Text
尝试过借助stop word想把真SQL的相似度降下来，但是0.05其实已经很低了，降不了多少，剩下的像"中"、"联"等词元都是恰好对上的
同理，真Text的相似度也没法提升
因此最后还是只能用基金、股票等关键字辅助判断
"""
