# -*- coding: utf-8 -*-
# @file label_question.py
# @author zhangshilong
# @date 2024/7/5

import pandas as pd

QUESTION_NUM = 1000
LABELS = ["Text", "SQL"]
WORKSPACE_DIR = "/mnt/workspace"
DATASET_DIR = f"{WORKSPACE_DIR}/bs_challenge_financial_14b_dataset"
VALIDATION_DIR = f"{WORKSPACE_DIR}/validation"

SAMPLE_NUM = 100
# 保证抽样的问题是固定的
SEED = 1024

question_df = pd.read_json(f"{DATASET_DIR}/question.json", lines=True)
assert len(question_df) == QUESTION_NUM

sample_df = question_df.sample(SAMPLE_NUM, random_state=SEED)
sample_df.sort_index(inplace=True)
sample_df.rename(columns={"id": "问题id", "question": "问题"}, inplace=True)
sample_df["标签"] = ""

# 导出后，人工填上标签保存为question_test.json
sample_df.to_json(f"{VALIDATION_DIR}/raw_question_test.json", orient="records", force_ascii=False, indent=4)

# 载入标好的数据
label_df = pd.read_json(f"{VALIDATION_DIR}/question_test.json")
assert len(label_df) == SAMPLE_NUM

label_counts = label_df["标签"].value_counts()
print(f"{label_counts=}")  # 57个SQL, 43个Text
assert set(label_counts.index.tolist()) == set(LABELS)

# 分类打印出来，复验是否标错
for label in LABELS:
    print(f"{label=}")
    for q in label_df.query(f"标签=='{label}'")["问题"]:
        print(f"\t{q}\n")
    print()
# 经复验，标签全部正确
"""
问题id 879（生态环境建设行业的上游是什么行业？） 看着像SQL类别，但是找不到答案。原作也是生成一个无法执行的sql语句
"""
