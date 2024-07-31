# -*- coding: utf-8 -*-
# @file verify_sql_prompt.py
# @author zhangshilong
# @date 2024/7/24

from ..tools.config import Config

tokenizer = Config.get_tokenizer()
model = Config.get_model()

answer_df = Config.get_sql_question_answer_df()
answer_questions = answer_df["问题"].tolist()

example_df = Config.get_sql_prompt_example_df()
example_questions = example_df["问题"].tolist()

# =====================================================================================
# 为每个问题找出最相似的n个样例


few_shot_num = 3
distance_matrix = tokenizer.pairwise_jaccard_distances(answer_questions, example_questions)
indices = distance_matrix.argsort(axis=1)[:, :few_shot_num]  # argsort只能升序，所以用距离，越小越相似

# =====================================================================================
# 构造prompt


prompt_template = """任务描述：
给你一条问题和查询结果，你需要参照示例的格式将它们整合成答案并输出。


示例：
{examples}


请参考上述示例格式，整合下面的问题和查询结果并输出答案。
问题: {question}
查询结果：{result}
答案："""


def make_example_prompt(record):
    question = record["问题"]
    result = record["SQL结果"]
    answer = record["答案"]
    return f"问题: {question}\n查询结果：{result}\n答案：{answer}"


def make_prompt(row):
    similar_df = example_df.iloc[indices[row.name]]
    examples = "\n\n".join([make_example_prompt(record) for _, record in similar_df.iterrows()])
    question = row["问题"]
    result = row["SQL结果"]
    prompt = prompt_template.format(examples=examples, question=question, result=result)
    return prompt


prompts = answer_df.progress_apply(make_prompt, axis=1).tolist()
# answer_df["prompt"] = prompts

print(prompts[0])

# =====================================================================================
# 批量推理


pred_answers = model.batch(tokenizer, prompts, system="你是一个擅长整合资料的助手。", batch_size=4)
answer_df["预测答案"] = pred_answers

answer_df["答案正确"] = answer_df["答案"] == answer_df["预测答案"]
question_num = len(answer_df)
correct_num = sum(answer_df["答案正确"])
print(f"测试问题数：{question_num}")  # 600
print(f"答案正确数：{correct_num}")
print(f"答案正确率：{correct_num / question_num:.2%}")
# 展示bad case
answer_df.query("答案正确 == False")


def is_contain_result(row):
    values = [str(value) for record in eval(row["SQL结果"]) for value in record.values()]
    return all([value in row["预测答案"] for value in values])


answer_df["包含结果"] = answer_df.progress_apply(is_contain_result, axis=1)
contain_num = sum(answer_df["包含结果"])
print(f"测试问题数：{question_num}")  # 600
print(f"包含结果数：{contain_num}")
print(f"包含结果率：{contain_num / question_num:.2%}")
# 展示bad case
answer_df.query("包含结果 == False")

"""
总结：600条问题，回答完全一致83.83%，但包含SQL结果的有99.00%

"""
