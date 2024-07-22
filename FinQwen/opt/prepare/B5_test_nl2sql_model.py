# -*- coding: utf-8 -*-
# @file B5_test_nl2sql_model.py
# @author zhangshilong
# @date 2024/7/22

from ..tools.config import Config
from ..tools.constant import ModelName

model_name = ModelName.QWEN_7B_CHAT
adapter_dir = "/mnt/workspace/prepare/output/nl2sql_lora/20240722_181022/checkpoint-1000"  # 这个adapter是跟7B搭配的
tokenizer = Config.get_tokenizer(model_name)
model = Config.get_model(model_name, adapter_dir=adapter_dir)

# =====================================================================================
# 用模型生成sql并执行

db = Config.get_database()
test_df = Config.get_sql_test_question_df()
test_df.drop(columns="答案", inplace=True)

questions = test_df["问题"].tolist()
true_sqls = test_df["SQL"].tolist()
true_results = test_df["SQL结果"].tolist()

pred_sqls = model.batch(tokenizer, questions, batch_size=4)  # TODO 待调节
test_df["预测SQL"] = pred_sqls

pred_results = [
    None if raw_result is None else str(raw_result.to_dict(orient="records"))
    for raw_result in db.batch_query(pred_sqls, raise_error=False)
]
test_df["预测SQL结果"] = pred_results

test_df["预测正确"] = test_df["SQL结果"] == test_df["预测SQL结果"]
question_num = len(test_df)
execute_num = sum(test_df["预测SQL结果"].notnull())
correct_num = sum(test_df["预测正确"])
print(f"测试问题数： {question_num}")
print(f"成功执行数：{execute_num}")
print(f"成功执行率：{execute_num / question_num:.2%}")
print(f"预测正确数：{correct_num}")
print(f"预测正确率：{correct_num / question_num:.2%}")
# 展示bad case
test_df.query("预测正确 == False")

# =====================================================================================
# TODO
"""
1.8B:
    loss为0.000478
    test前10个只有5个能执行，而且只有4个答案对了

7B:
    loss为0.000351
    test前10个只有9个能执行，而且只有8个答案对了
    
    全量：
        bs=16: 爆内存
        bs=4: 利用率86-92%, 显存18512MB
    
"""
