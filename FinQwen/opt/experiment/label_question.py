# -*- coding: utf-8 -*-
# @file label_question.py
# @author zhangshilong
# @date 2024/7/7

import pandas as pd

from ..tools.config import Config
from ..tools.constant import Label
from ..tools.utils import File

question_df = Config.get_question_df()

# 虽然Config.set_seed()已经保证了第一次运行这句必然有相同的结果，但考虑到在notebook可能会多次运行，因此还是加上seed
sample_df = question_df.sample(Config.TEST_QUESTION_NUM, random_state=Config.SEED)
sample_df.sort_index(inplace=True)
sample_df["标签"] = ""

# 导出成json格式，方便填上标签
sample_df.to_json(f"{Config.EXPERIMENT_OUTPUT_DIR}/raw_test_question.json", orient="records", force_ascii=False,
                  indent=4)

# =====================================================================
# 人工填上标签后另存为test_question.json, 放回experiment/output文件夹
# =====================================================================

# 载入标好的数据
test_question_df = pd.read_json(f"{Config.EXPERIMENT_OUTPUT_DIR}/test_question.json")

# 查看统计结果
label_counts = test_question_df["标签"].value_counts()
print(f"{label_counts=}")  # 57个SQL, 43个Text
assert set(label_counts.index.tolist()) == Label.values()

# 分类打印出来，复验是否标错
for label, df in test_question_df.groupby("标签"):
    print(f"{label=}")
    for question in df["问题"]:
        print(f"\t{question}\n")
    print()
# 经复验，标签全部正确


# 载入后续调试过程中发现的bad case
classification_bad_case = File.json_load(f"{Config.EXPERIMENT_OUTPUT_DIR}/classification_bad_case.json")

# 合并到test_question_df
for label, bad_cases in classification_bad_case.items():
    bad_case_df = pd.DataFrame(bad_cases)
    bad_case_df.rename(columns={"id": "问题id", "question": "问题"}, inplace=True)
    bad_case_df["标签"] = label

    # 如果数据出现重复，必须有相同的标签
    qids = bad_case_df["问题id"].tolist()
    assert all(test_question_df.query(f"问题id in {qids}")["标签"] == label)
    # 会自动去重 + 排序
    test_question_df = pd.merge(test_question_df, bad_case_df, how="outer", copy=False)

# 保存到intermediate文件夹
test_question_df.to_csv(Config.TEST_QUESTION_PATH, index=False)

"""
结论：除了id=879（生态环境建设行业的上游是什么行业？）在文档和数据库都找不到答案，其它均标记无误
将其标为SQL吧

bad case补充了10个Text
"""
