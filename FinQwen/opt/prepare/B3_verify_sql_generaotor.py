# -*- coding: utf-8 -*-
# @file B3_verify_sql_generaotor.py
# @author zhangshilong
# @date 2024/7/19

from ..tools.sql_generator import Manager

# 先人工为57个聚类的问题编写对应的Generator, 并利用export方法校验正确性

# 输出整体校验结果
Manager.analysis()

# 生成600条SQL问题的submit_result.jsonl，耗时10:36
Manager.export()
