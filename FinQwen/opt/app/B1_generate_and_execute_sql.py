# -*- coding: utf-8 -*-
# @file B1_generate_and_execute_sql.py
# @author zhangshilong
# @date 2024/7/24

from ..tools.config import Config
from ..tools.constant import Category
from ..tools.constant import ModelName
from ..tools.utils import File

db = Config.get_database()

model_name = ModelName.QWEN_7B_CHAT
tokenizer = Config.get_tokenizer(model_name)

adapter_dir = Config.nl2sql_adapter_dir(version="20240724_165829", step=950)  # 最好的checkpoint
model = Config.get_model(model_name, adapter_dir=adapter_dir)

question_classification_df = Config.get_question_classification_df()
question_df = question_classification_df.query(f"问题分类 == '{Category.SQL}'").reset_index()[
    ["问题id", "问题"]]  # 600条
questions = question_df["问题"].tolist()

# 生成sql，耗时11:11
sqls = model.batch(tokenizer, questions, batch_size=8)
question_df["SQL"] = sqls

# 执行sql，耗时5:27
results = [
    None if raw_result is None else raw_result.to_dict(orient="records")
    for raw_result in db.batch_query(sqls, raise_error=False)
]
question_df["SQL结果"] = results

# 保存
File.dataframe_to_csv(question_df, Config.SQL_RESULT_PREDICTION_PATH)
