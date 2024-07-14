# -*- coding: utf-8 -*-
# @file 1_generate_raw_classification_test_question.py
# @author zhangshilong
# @date 2024/7/12

from ..tools.config import Config

question_df = Config.get_question_df()

# 虽然Config.set_seed()已经保证了第一次运行这句必然有相同的结果，但考虑到在notebook可能会多次运行，因此还是加上seed
sample_df = question_df.sample(Config.CLASSIFICATION_TEST_QUESTION_SAMPLE_NUM, random_state=Config.SEED)
sample_df.sort_index(inplace=True)
sample_df["问题分类标签"] = ""
sample_df["公司名称标签"] = None

# 导出成json格式，方便填上标签
sample_df.to_json(f"{Config.PREPARE_OUTPUT_DIR}/raw_classification_test_question.json", orient="records",
                  force_ascii=False, indent=4)

# 人工填上标签后另存为classification_test_question.json, 放回prepare/output文件夹
