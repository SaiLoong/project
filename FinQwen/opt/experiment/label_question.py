# -*- coding: utf-8 -*-
# @file label_question.py
# @author zhangshilong
# @date 2024/7/7

import pandas as pd

from ..tools.config import Config
from ..tools.constant import Label

question_df = Config.get_question_df()

# 虽然Config.set_seed()已经保证了第一次运行这句必然有相同的结果，但考虑到在notebook可能会多次运行，因此还是加上seed
sample_df = question_df.sample(Config.TEST_QUESTION_NUM, random_state=Config.SEED)
sample_df.sort_index(inplace=True)
sample_df["标签"] = ""

# 导出成json格式，方便填上标签
sample_df.to_json(f"{Config.EXPERIMENT_DIR}/raw_test_question.json", orient="records", force_ascii=False, indent=4)

# 人工填上标签后另存为test_question.json, 放回experiment文件夹

# 载入标好的数据
test_question_df = pd.read_json(f"{Config.EXPERIMENT_DIR}/test_question.json")

# 查看统计结果
label_counts = test_question_df["标签"].value_counts()
print(f"{label_counts=}")  # 57个SQL, 43个Text
assert set(label_counts.index.tolist()) == Label.values()

# 分类打印出来，复验是否标错
for label in Label.values():
    print(f"{label=}")
    for question in test_question_df.query(f"标签 == '{label}'")["问题"]:
        print(f"\t{question}\n")
    print()
# 经复验，标签全部正确

# 保存到intermediate文件夹
test_question_df.to_csv(Config.TEST_QUESTION_PATH, index=False)

"""
结论：除了id=879（生态环境建设行业的上游是什么行业？）在文档和数据库都找不到答案，其它均标记无误
将其标为SQL吧
"""
