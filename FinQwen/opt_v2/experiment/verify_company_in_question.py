# -*- coding: utf-8 -*-
# @file verify_company_in_question.py
# @author zhangshilong
# @date 2024/7/7

import re

from ..tools.config import Config

# 获取问题
question_df = Config.get_question_df()
questions = question_df["question"].tolist()

# 获取公司名
ref_company_df = Config.get_ref_company_df()
companies = ref_company_df["公司名称"].tolist()

# 逐个匹配
found_dict = {company: [] for company in companies}
unfound_list = []
pattern = "|".join(companies)
for question in questions:
    if m := re.search(pattern, question):
        found_dict[m.group()].append(question)
    else:
        unfound_list.append(question)

# 初步分析
print(f"不包含公司名的问题数量：{len(unfound_list)}\n")  # 611个
# 没有在问题出现过的公司
companies2 = [company for company, qs in found_dict.items() if not qs]
print(f"没有在问题出现过的公司：{len(companies2)}")  # 39间
for company in companies2:
    print(f"\t{company}")

# 去掉"股份有限公司"再匹配
companies3 = [company.rsplit("股份有限公司", 1)[0] for company in companies2]
for company in companies3:
    print(f"\t{company}")
# 再次确认真的不存在
for company in companies3:
    for question in questions:
        if company in question:
            print(f"{company}  in  {question}")

# 人工复核
for question in unfound_list:
    print(f"{question}\n")

"""
结论：当前公司名并没有什么问题。只需把 沈阳晶格自动化技术有限公司 -> 深圳麦格米特电气股份有限公司 即可
"""
