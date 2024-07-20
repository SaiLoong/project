# -*- coding: utf-8 -*-
# @file B4_finetune_NL2SQL_model.py
# @author zhangshilong
# @date 2024/7/19

from ..tools.config import Config

train_df, val_df, test_df = Config.get_sql_question_dfs()

# TODO
#  1. 生成完整的sql答案，看分数
#  2. 重新生成数据集（可能要好几个小时，主要是28a、28b很慢）
#    - 进度条弄回内部刷新的吧，外部28卡太久了
#    - 如果完整sql答案没有变化，那就不用重新生成了
#  3. 写微调脚本
#    - 回忆一下官方和之前demo的笔记，区别的地方 可以对比微调效果看谁更有效
#    - 多看看官方文档和提供的finetune脚本
