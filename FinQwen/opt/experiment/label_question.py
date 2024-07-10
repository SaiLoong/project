# -*- coding: utf-8 -*-
# @file label_question.py
# @author zhangshilong
# @date 2024/7/7

import numpy as np
import pandas as pd

from ..tools.config import Config
from ..tools.constant import Label
from ..tools.utils import File

question_df = Config.get_question_df()

# 虽然Config.set_seed()已经保证了第一次运行这句必然有相同的结果，但考虑到在notebook可能会多次运行，因此还是加上seed
sample_df = question_df.sample(Config.SAMPLE_QUESTION_NUM, random_state=Config.SEED)
sample_df.sort_index(inplace=True)
sample_df["问题标签"] = ""
sample_df["公司名称"] = None
sample_df["数据库表名"] = None

# 导出成json格式，方便填上标签
sample_df.to_json(f"{Config.EXPERIMENT_OUTPUT_DIR}/raw_test_question.json", orient="records", force_ascii=False,
                  indent=4)

# =====================================================================


# 人工填上标签后另存为test_question.json, 放回experiment/output文件夹


# =====================================================================


# 载入标好的数据
test_question_df = pd.read_json(f"{Config.EXPERIMENT_OUTPUT_DIR}/test_question.json")
# 对于null，公司名称列转化为None，而数据库表名列转化为nan，统一为nan方便后续处理
test_question_df.fillna(np.nan, inplace=True)

company_df, companies = Config.get_company_df(return_companies=True)


# 校验有无缺标、标错
def func(row):
    label = row["问题标签"]
    company = row["公司名称"]
    tables = row["数据库表名"]
    assert label in Label.values()
    if label == Label.TEXT:
        assert company in companies
    else:
        assert company != company  # np.nan好迷惑的性质
    # TODO 暂时先不填数据库部分，之后再填坑
    assert tables != tables


test_question_df.progress_apply(func, axis=1)

# 查看统计结果
label_counts = test_question_df["问题标签"].value_counts()
print(f"{label_counts=}")  # 57个SQL, 43个Text

# 分类打印出来，复验是否标错
for label, df in test_question_df.groupby("问题标签"):
    print(f"{label=}")
    for _, row in df.iterrows():
        question = row["问题"]
        company = row["公司名称"]
        tables = row["数据库表名"]
        print(f"\t{question}")
        if label == Label.TEXT:
            print(f"\t\t{company}")
        else:
            print(f"\t\t{tables}")
        print()
    print()
# 经复验，标签全部正确


# =====================================================================


# 载入后续调试过程中发现的bad case
classification_bad_case = File.json_load(f"{Config.EXPERIMENT_OUTPUT_DIR}/classification_bad_case.json")

# 合并到test_question_df
for label, bad_cases in classification_bad_case.items():
    # 校验值的合法性
    assert label in Label.values()
    bad_case_df = pd.DataFrame(bad_cases)
    if label == Label.TEXT:
        assert set(bad_case_df["公司名称"].tolist()).issubset(companies)
    else:
        pass  # TODO 暂时先不填数据库部分，之后再填坑

    bad_case_df["问题标签"] = label
    # 如果数据出现重复，必须有相同的标签
    qids = bad_case_df["问题id"].tolist()
    compare_df = test_question_df.query(f"问题id in {qids}")["问题标签"] == label
    assert all(compare_df)
    print(f"添加{len(qids) - len(compare_df)}条{label}分类的样本: {qids}")
    # 会自动去重 + 排序，缺失的列填上nan
    test_question_df = pd.merge(test_question_df, bad_case_df, how="outer", copy=False)

# 添加公司id
company_to_cid_mapping = company_df.set_index(keys="公司名称")["公司id"].to_dict()
test_question_df["公司id"] = test_question_df["公司名称"].map(company_to_cid_mapping)
test_question_df = test_question_df.reindex(columns=["问题id", "问题", "问题标签", "公司名称", "公司id", "数据库表名"])

# 保存到intermediate文件夹
test_question_df.to_csv(Config.TEST_QUESTION_PATH, index=False)

"""
结论：除了id=879（生态环境建设行业的上游是什么行业？）在文档和数据库都找不到答案，其它均标记无误
将其标为SQL吧

bad case补充了10个Text
"""
