# -*- coding: utf-8 -*-
# @file B5_evaluate_nl2sql_model.py
# @author zhangshilong
# @date 2024/7/22

from ..tools.config import Config
from ..tools.constant import ModelName
from ..tools.utils import File

model_name = ModelName.QWEN_7B_CHAT
tokenizer = Config.get_tokenizer(model_name)

adapter_dir = Config.nl2sql_adapter_dir(version="20240724_041234", step=600)
model = Config.get_model(model_name, adapter_dir=adapter_dir)

# =====================================================================================


db = Config.get_database()
answer_df = Config.get_sql_question_answer_df()

pred_df = answer_df.drop(columns=["问题聚类", "答案"])
question_num = len(pred_df)
questions = pred_df["问题"].tolist()
# true_sqls = pred_df["SQL"].tolist()
# true_results = pred_df["SQL结果"].tolist()


# =====================================================================================
# 预测sql


pred_sqls = model.batch(tokenizer, questions, batch_size=8)
pred_df["预测SQL"] = pred_sqls

pred_df["SQL正确"] = pred_df["SQL"] == pred_df["预测SQL"]
sql_correct_num = sum(pred_df["SQL正确"])
print(f"测试问题数：{question_num}")  # 600
print(f"SQL正确数：{sql_correct_num}")
print(f"SQL正确率：{sql_correct_num / question_num:.2%}")
# 展示bad case
pred_df.query("SQL正确 == False")

# =====================================================================================
# 执行sql


pred_results = [
    None if raw_result is None else str(raw_result.to_dict(orient="records"))
    for raw_result in db.batch_query(pred_sqls, raise_error=False)
]
pred_df["预测SQL结果"] = pred_results

pred_df["结果正确"] = pred_df["SQL结果"] == pred_df["预测SQL结果"]
execute_num = sum(pred_df["预测SQL结果"].notnull())
result_correct_num = sum(pred_df["结果正确"])
print(f"测试问题数：{question_num}")  # 600
print(f"成功执行数：{execute_num}")
print(f"成功执行率：{execute_num / question_num:.2%}")
print(f"结果正确数：{result_correct_num}")
print(f"结果正确率：{result_correct_num / question_num:.2%}")
print(f"结果正确/成功执行：{result_correct_num / execute_num:.2%}")
# 展示bad case
pred_df.query("结果正确 == False")

# =====================================================================================


# 保存下来，不要浪费
File.dataframe_to_csv(pred_df, f"{adapter_dir}/nl2sql_evaluate.csv")
