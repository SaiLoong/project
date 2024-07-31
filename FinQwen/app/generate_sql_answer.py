# -*- coding: utf-8 -*-
# @file generate_sql_answer.py
# @author zhangshilong
# @date 2024/7/24

import pandas as pd

from ..tools.config import Config
from ..tools.utils import File

# 用Tongyi-Finance-14B-Chat-Int4
tokenizer = Config.get_tokenizer()
model = Config.get_model()

prediction_df = Config.get_sql_result_prediction_df()
example_df = Config.get_sql_prompt_example_df()

# 预测结果为空的，答案直接拷贝问题
invalid_df = prediction_df.query("SQL结果.isna()").reset_index(drop=True)
print(f"有{len(invalid_df)}个问题的SQL结果为None")
invalid_df["答案"] = invalid_df["问题"]

# 预测结果不为空的，为每个问题找出最相似的n个样例，few-shot prompting
valid_df = prediction_df.query("SQL结果.notna()").reset_index(drop=True)
valid_questions = valid_df["问题"].tolist()
example_questions = example_df["问题"].tolist()

few_shot_num = 3
distance_matrix = tokenizer.pairwise_jaccard_distances(valid_questions, example_questions)
indices = distance_matrix.argsort(axis=1)[:, :few_shot_num]  # argsort只能升序，所以用距离，越小越相似

# 构造prompt
prompt_template = """任务描述：
给你一条问题和查询结果，你需要参照示例的格式将它们整合成答案并输出。


示例：
{examples}


请参考上述示例格式，整合下面的问题和查询结果并输出答案。
问题: {question}
查询结果：{result}
答案："""


def make_example_prompt(example):
    question = example["问题"]
    result = example["SQL结果"]
    answer = example["答案"]
    return f"问题: {question}\n查询结果：{result}\n答案：{answer}"


def make_prompt(row):
    similar_df = example_df.iloc[indices[row.name]]
    examples = "\n\n".join([make_example_prompt(record) for _, record in similar_df.iterrows()])
    question = row["问题"]
    result = row["SQL结果"]
    prompt = prompt_template.format(examples=examples, question=question, result=result)
    return prompt


prompts = valid_df.progress_apply(make_prompt, axis=1).tolist()
print(prompts[-22])

# 批量推理，耗时13:48
answers = model.batch(tokenizer, prompts, system="你是一个擅长整合资料的助手。", batch_size=4)
valid_df["答案"] = answers

# 合并答案、保存并生成submit_result.jsonl
df = pd.concat([invalid_df, valid_df])
df.sort_values(by="问题id", inplace=True)
df.reset_index(drop=True, inplace=True)

File.dataframe_to_csv(df, Config.SQL_ANSWER_PREDICTION_PATH)
Config.export_submit_result(df, Config.SQL_SUBMIT_RESULT_PATH)  # 95.00
