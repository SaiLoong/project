# -*- coding: utf-8 -*-
# @file finetune_nl2sql_model.py
# @author zhangshilong
# @date 2024/7/19

import os
import warnings

from peft import get_peft_model
from peft import LoraConfig
from peft import TaskType
from transformers import DataCollatorForSeq2Seq
from transformers import EarlyStoppingCallback
from transformers import Trainer
from transformers import TrainingArguments

from ..tools.config import Config
from ..tools.constant import ModelMode
from ..tools.constant import ModelName
from ..tools.utils import File
from ..tools.utils import Time

# 這些警告已确认与用户使用无关，大多是Qwen的问题
warnings.filterwarnings("ignore",
                        message="Passing the following arguments to `Accelerator` is deprecated")  # transformers的使用问题
warnings.filterwarnings("ignore", message=".*the use_reentrant parameter should be passed explicitly")  # Qwen的使用问题
warnings.filterwarnings("ignore", message="Could not find a config file")  # 不懂为啥保存时需要和远程仓库校对词汇表

# ================================================================================================
# 模型部分


model_name = ModelName.QWEN_7B_CHAT
mode = ModelMode.TRAIN
tokenizer = Config.get_tokenizer(model_name, mode=mode)

model = Config.get_model(model_name, mode=mode)
lora_config = LoraConfig(
    # 不加的话最后的model是PeftModel，加了是PeftModelForCausalLM，看代码貌似没有什么关键改动，但是以防万一加上吧
    task_type=TaskType.CAUSAL_LM,

    # finetune
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,

    # demo
    # r=8,
    # lora_alpha=32,
    # lora_dropout=0.1,
    target_modules=["c_attn", "c_proj", "w1", "w2"]
)

model = get_peft_model(model, lora_config)
model.enable_input_require_grads()  # peft与gradient checkpoint同时用时必须加
model.print_trainable_parameters()

# ================================================================================================
# 数据部分

system = "你是一名Mysql数据库开发人员。"
prompt_template = """任务描述：
你精通Mysql数据库的sql代码编写，你需要根据给定的问题编写sql代码。
为了确保sql代码可执行，你必须保证sql代码语法没有错误（例如左右括号数量一致）。

问题：{question}
sql代码："""


def func(example):
    question = example["问题"]
    sql = example["SQL"]
    prompt = prompt_template.format(question=question)
    return tokenizer.make_finetune_inputs(prompt, sql, system=system)


dataset = Config.get_nl2sql_dataset_v2()
dataset = dataset.map(func, remove_columns=dataset.column_names["train"], num_proc=os.cpu_count())
train_dataset = dataset["train"]  # 10000
validation_dataset = dataset["validation"]  # 1000
test_dataset = dataset["test"]  # 1000

# ================================================================================================
# 训练部分

# 按训练时间分文件夹，避免覆盖之前训练的模型
fmt = "%Y%m%d_%H%M%S"
output_dir = f"{Config.NL2SQL_FINETUNE_DIR}/{Time.current_time(fmt)}"
print(f"{output_dir=}")

training_args = TrainingArguments(
    output_dir=output_dir,
    seed=Config.SEED,

    # 训练
    num_train_epochs=4,  # 有early stop
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,

    # 验证
    evaluation_strategy="steps",
    eval_steps=50,
    per_device_eval_batch_size=16,

    # 日志
    # logging_strategy="steps",  # 默认就是这个
    logging_steps=50,  # 使用notebook时，如果开启了eval，只有进行eval的step才会打印日志

    # 保存
    # save_strategy="steps",  # 默认就是这个
    save_steps=50,  # 和eval保持一致
    save_total_limit=2,
    load_best_model_at_end=True,

    # 超参数
    learning_rate=3e-4,
    weight_decay=0.1,
    adam_beta2=0.95,
    warmup_ratio=0.01,
    lr_scheduler_type="cosine"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=validation_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
)

trainer.train()

# trainer.save_model(f"{output_dir}/best")


# ================================================================================================
# 测试部分


trainer.evaluate(test_dataset)

# ================================================================================================
# ================================================================================================
# 直接执行B5的逻辑


db = Config.get_database()
answer_df = Config.get_sql_question_answer_df()

pred_df = answer_df.drop(columns=["问题聚类", "答案"])
question_num = len(pred_df)
questions = pred_df["问题"].tolist()
# true_sqls = pred_df["SQL"].tolist()
# true_results = pred_df["SQL结果"].tolist()


# =====================================================================================
# 预测sql

tokenizer.padding_side = "left"  # 批量推理必须为左填充
prompts = [prompt_template.format(question=question) for question in questions]
print(prompts[0])

pred_sqls = model.batch(tokenizer, prompts, system=system, batch_size=8)
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
File.dataframe_to_csv(pred_df, f"{output_dir}/nl2sql_evaluate.csv")
