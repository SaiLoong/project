# -*- coding: utf-8 -*-
# @file 2_generate_classification_test_question.py
# @author zhangshilong
# @date 2024/7/12

import numpy as np
import pandas as pd

from ..tools.config import Config
from ..tools.constant import Category
from ..tools.utils import File

# 载入标好的数据
test_question_df = pd.read_json(f"{Config.PREPARE_OUTPUT_DIR}/classification_test_question.json")
assert len(test_question_df) == Config.CLASSIFICATION_TEST_QUESTION_SAMPLE_NUM
# 将None转化为np.nan，方便后续处理
test_question_df.fillna(np.nan, inplace=True)

company_df, companies = Config.get_company_df(return_companies=True)


# 校验有无缺标、标错
def func(row):
    category = row["问题分类标签"]
    company = row["公司名称标签"]
    assert category in Category.values()
    if category == Category.TEXT:
        assert company in companies
    else:
        assert company != company  # np.nan好迷惑的性质


test_question_df.progress_apply(func, axis=1)

# 查看统计结果
category_counts = test_question_df["问题分类标签"].value_counts()
print(f"{category_counts=}")  # 56个SQL, 44个Text

# 分类打印出来，复验是否标错
for category, df in test_question_df.groupby("问题分类标签"):
    print(f"{category=}")
    for _, row in df.iterrows():
        question = row["问题"]
        company = row["公司名称标签"]
        print(f"\t{question}")
        if category == Category.TEXT:
            print(f"\t\t{company}")
        print()
    print()
# 经复验，标签全部正确


# =====================================================================


# 载入后续调试过程中发现的bad case
classification_bad_case = File.json_load(f"{Config.PREPARE_OUTPUT_DIR}/classification_bad_case.json")

# 合并到test_question_df
for category, bad_cases in classification_bad_case.items():
    # 校验值的合法性
    assert category in Category.values()
    bad_case_df = pd.DataFrame(bad_cases)
    if category == Category.TEXT:
        assert set(bad_case_df["公司名称标签"].tolist()).issubset(companies)

    bad_case_df["问题分类标签"] = category
    # 如果数据出现重复，必须有相同的标签
    qids = bad_case_df["问题id"].tolist()
    compare_df = test_question_df.query(f"问题id in {qids}")["问题分类标签"] == category
    assert all(compare_df)
    print(f"添加{len(qids) - len(compare_df)}条{category}分类的样本: {qids}")
    # 会自动去重 + 排序，缺失的列填上nan
    test_question_df = pd.merge(test_question_df, bad_case_df, how="outer", copy=False)

# 添加公司id
company_to_cid_mapping = company_df.set_index(keys="公司名称")["公司id"].to_dict()
test_question_df["公司id标签"] = test_question_df["公司名称标签"].map(company_to_cid_mapping)

# 保存到intermediate文件夹
File.dataframe_to_csv(test_question_df, Config.CLASSIFICATION_TEST_QUESTION_PATH)
