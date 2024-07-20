# -*- coding: utf-8 -*-
# @file B3_generate_sql_dataset.py
# @author zhangshilong
# @date 2024/7/19

from ..tools.sql_generator import Manager

# 先人工为57个聚类的问题编写对应的Generator放到tools.sql_generator.py, 并利用export方法校验正确性

# 输出整体校验结果
Manager.analysis()
"""
data_query预计得分: 97.24

丢分超过0.1的Generator:
[4] 4.38/4.5
[11] 1.55/1.67
[12] 3.21/3.33
[17] 3.16/3.33
[21] 1.21/1.67
[28a] 1.56/1.67
[28b] 0.06/0.17
[35b] 0.06/0.17
[40] 1.49/1.67
"""
# 实测97.22

# 生成600条SQL问题的submit_result.jsonl，耗时10:20
Manager.export()

# TODO 数字是否确认为这个？
# 生成sql数据集，耗时约7h左右（主要是28a、28b太久了）
train_df, val_df, test_df = Manager.generate_dataset(10000, 1000, 1000)
