# -*- coding: utf-8 -*-
# @file verify_ref_label.py
# @author zhangshilong
# @date 2024/7/5

import pandas as pd

QUESTION_NUM = 1000
SAMPLE_NUM = 100
WORKSPACE_DIR = "/mnt/workspace"
VALIDATION_DIR = f"{WORKSPACE_DIR}/validation"

# 先将ref的A01_question_classify.csv拷贝至VALIDATION_DIR
# 为了不引起歧义，改名为ref_A01_question_classify.csv
ref_df = pd.read_csv(f"{VALIDATION_DIR}/ref_A01_question_classify.csv")
assert len(ref_df) == QUESTION_NUM

# 加载人工标注数据
label_df = pd.read_json(f"{VALIDATION_DIR}/question_test.json")
assert len(label_df) == SAMPLE_NUM

ref_df2 = ref_df[["问题id", "分类"]]
merge_df = pd.merge(label_df, ref_df2, how="left", on="问题id")

merge_df["correct"] = merge_df["标签"] == merge_df["分类"]
merge_df["correct"].sum()
merge_df.query("correct == False")
"""
正确99条

id=659, "2019年中期报告里，国融基金管理有限公司管理的基金中，机构投资者持有份额比个人投资者多的基金有多少只?"
应为SQL, ref预测为Text

select count(*)
from 基金份额持有人结构 t1
join 基金基本信息 t2
on t1.[基金代码] = t2.[基金代码]
where t1.[机构投资者持有的基金份额] > t1.[个人投资者持有的基金份额]
    and t1.[定期报告所属年度] = '2019'
    and t1.[报告类型] = '中期报告'
    and t2.[管理人] = '国融基金管理有限公司'
limit 1;
"""
