# -*- coding: utf-8 -*-
# @file B4_finetune_nl2sql_model.py
# @author zhangshilong
# @date 2024/7/19

import warnings

from datasets import load_dataset
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
from ..tools.nl2sql_utils import build_finetune_data
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
    # 之前demo的参数是r=8, lora_alpha=32, lora_dropout=0.1
    r=64,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["c_attn", "c_proj", "w1", "w2"]
)

model = get_peft_model(model, lora_config)
model.enable_input_require_grads()  # peft与gradient checkpoint同时用时必须加
model.print_trainable_parameters()

# ================================================================================================
# 数据部分


ds = load_dataset("csv", data_files={
    "train": Config.SQL_TRAIN_QUESTION_PATH,  # 10000
    "validation": Config.SQL_VALIDATION_QUESTION_PATH,  # 1000
    "test": Config.SQL_TEST_QUESTION_PATH  # 1000
})


def func(example):
    question = example["问题"]
    sql = example["SQL"]
    return build_finetune_data(tokenizer, question, sql)


dataset = ds.map(func, remove_columns=ds.column_names["train"], num_proc=4)
train_dataset = dataset["train"]
validation_dataset = dataset["validation"]
test_dataset = dataset["test"]

# ================================================================================================
# 训练部分

# 按训练时间分文件夹，避免覆盖之前训练的模型
fmt = "%Y%m%d_%H%M%S"
output_dir = f"{Config.PREPARE_OUTPUT_DIR}/nl2sql_lora/{Time.current_time(fmt)}"
print(f"{output_dir=}")

training_args = TrainingArguments(
    output_dir=output_dir,
    seed=Config.SEED,

    # 训练
    num_train_epochs=10,  # 反正有early stop
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
    save_total_limit=4,
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
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)]
)

trainer.train()

trainer.save_model(f"{output_dir}/best")

# 将挑选好checkpoint拷贝至intermediate/adapter/nl2sql


# ================================================================================================
# 测试部分

trainer.evaluate(test_dataset)

# TODO
"""
1.8B 20240722_151706
    bs=16，在2450 step验证集最小(0.000256，约4w样本，4epoch)，总耗时1:09:22
    显存22348M，接近爆了；GPU利用率稳定在98%左右

7B 20240722_181022
    bs=4*4，在1000 step验证集最小(0.000287，约1.6w样本，1.6epoch)，总耗时1:38:11
   20240722_200616
    lora参数改小了，1650 step, 0.000274, 2:46:10
"""
