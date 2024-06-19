# -*- coding: utf-8 -*-
# @file train_lora.py
# @author zhangshilong
# @date 2024/6/15

import json
import os

import torch

# 保存模型时peft会访问huggingface，对比线上词汇表和本地的是否一致（如果不一致则也保存embedding参数）。但是DSW连接不上，因此换成国内镜像站
# 必须放在任何huggingface系列库的前面
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import datasets
from peft import get_peft_model
from peft import LoraConfig
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq
from transformers import Trainer
from transformers import TrainingArguments

# 训练集共1949972条，只取前1000条
with open("medical_zh/train_zh_0.jsonl", "r") as f:
    lst = [json.loads(next(f)) for _ in range(1000)]
"""
lst[1]=

{'instruction': '帕金森叠加综合征的辅助治疗有些什么？',
 'input': '',
 'output': '综合治疗；康复训练；生活护理指导；低频重复经颅磁刺激治疗'}
"""

# 存起来
with open("train_lora.json", "w") as f:
    json.dump(lst, f, ensure_ascii=False)  # 有汉字，加ensure_ascii=False

# 加载Dataset。原文经过DataFrame中转，比较麻烦
ds = datasets.load_dataset("json", data_files="train_lora.json", split="train")  # 不加split会返回DatasetDict
"""
ds[1]=

{'instruction': '帕金森叠加综合征的辅助治疗有些什么？',
 'input': '',
 'output': '综合治疗；康复训练；生活护理指导；低频重复经颅磁刺激治疗'}
"""

# 加载tokenizer
# 凡是from_pretrained本地的模型/分词器都要加trust_remote_code=True
# 原文加了use_fast=False。实测无论填True/False还是忽略，tokenizer.is_fast都是False，文档也没解释这个参数
# t5则是use_fast输什么，is_fast都是True
CKPT_PATH = "Qwen-1_8B-Chat"
tokenizer = AutoTokenizer.from_pretrained(CKPT_PATH, trust_remote_code=True)
# Qwen只定义了3个特殊词元：eod、im_start、im_end，基类预留的特殊词元都没定义
# 但后续需要padding，因此把eod当成pad使用（很多人都这么搞）
tokenizer.pad_token_id = tokenizer.eod_id
"""
QWenTokenizer(name_or_path='Qwen-1_8B-Chat', vocab_size=151851, model_max_length=8192, is_fast=False, padding_side='right', truncation_side='right', special_tokens={'pad_token': '<|endoftext|>'}, clean_up_tokenization_spaces=True),  added_tokens_decoder={}
"""


def process_func(example):
    # Qwen 1.8B/7B的词元输入上限是8k/32k，原文说限制为512，考虑到汉字（生僻字）可能会切分成多个词元，因此进一步缩短为384（75%）
    # 但是MAX_LENGTH后面是用来处理词元而不是汉字的！简单看成进一步限制长度吧
    MAX_LENGTH = 384

    # 下面两句将instruction和response先改写为prompt格式，再转化为词元id（可以用tokenizer.decode()将input_ids解码回文本）
    # prompt格式参考自Qwen模型的make_context函数，特别留意上一段的<|im_end|>与下一段的<|im_start|>之间必有换行符
    # 原文instruction末尾没有加换行符，response加了，可能是作者粗心了。现统一加上，并简化拼接公式
    """
    <|im_start|>system
    你是一个医学助手，需要回答用户关于医学的问题：<|im_end|>
    <|im_start|>user
    帕金森叠加综合征的辅助治疗有些什么？<|im_end|>

    """
    # 原文加了add_special_tokens=False，本意是希望tokenizer不要补充特殊字符。但是Qwen并没有实现这一功能，因此可以去掉
    instruction = tokenizer(
        "<|im_start|>system\n" +
        "你是一个医学助手，需要回答用户关于医学的问题：<|im_end|>\n" +
        "<|im_start|>user\n" +
        example["instruction"] + example["input"] + "<|im_end|>\n"
    )
    """
    <|im_start|>assistant
    综合治疗；康复训练；生活护理指导；低频重复经颅磁刺激治疗<|im_end|>
    
    """
    response = tokenizer(
        "<|im_start|>assistant\n" +
        example["output"] + "<|im_end|>\n"
    )

    """
    假设instruction的词元是[A1, A2, A3]、response的词元是[B1, B2, B3]，拼成模型输入并假设后面加两个<pad>，得到：
        input_ids: [A1, A2, A3,  B1, B2, B3,  <e>,  <pad>, <pad>]
        attention_mask: [1, 1, 1,  1, 1, 1,  1,  0, 0]
        labels: [-100, -100, -100,  B1, B2, B3,  <e>,  -100, -100]
        当然有可能不用padding甚至截断（没有<e>了），但padding是最复杂的，讨论这个
    input_ids经过attention时，除了要考虑attention_mask，还要考虑causal_mask
        例如B1只能使用[A1, A2, A3,  B1]、 <pad>只能使用[A1, A2, A3,  B1, B2, B3,  <e>]
    经过若干层attention后，得到每个词元的预测词元logits分布
    在计算loss前会进行shift操作，即logits删掉最后一个词元、labels删掉第一个词元，导致A3预测B1、B3预测<e>，符合LM建模规则
        labels前面的-100表示不考虑instruction内部的预测loss、后面的-100表示不考虑<e>以及<pad>的预测loss
        A1、A2最后的logits没有用上，但是在中间的attention层需要被其它词元用到
        <pad>最后的logits没有用上，在中间的attention层也没有其它词元需要它，但是为了矩阵乘法，只能顺便把它算出来
        <e>最后的logits没有用上，在中间的attention层也没有有意义的词元需要它，感觉attention_mask设为0好像也没问题
    """
    # 末尾要加上结束词元
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.eod_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    # 与input_ids相比，只是将instruction的词元id替换成-100，表示不计算loss
    # 虽然现在词元对应的是自己，但模型会进行shift，保证预测下一个词元
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.eod_id]

    # 截断
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


# 类似DataFrame的用法，并行处理数据
train_dataset = ds.map(process_func, remove_columns=ds.column_names)  # 去掉原始的列
"""
Dataset({
    features: ['input_ids', 'attention_mask', 'labels'],
    num_rows: 1000
})
"""

# load_in_8bit将除了lm_head的Linear层换成bnb.nn.Linear8bit层，要用此功能必须加载到GPU
# Qwen模型使用bf16混合精度训练、存储；原文可能用的gpu不支持bf16，所以指定torch_dtype=torch.half
# 本来只要GPU支持bf16，即使不设置参数bf16/torch_dtype，也会自动推断使用bf16
# 但是当使用量化后，必须要使用torch_dtype明确指定才会使用bf16，设置参数bf16=True也没用（详见Qwen的源码阅读笔记）
# 原文设置device_map="auto"，就是默认值，省略掉
model = AutoModelForCausalLM.from_pretrained(CKPT_PATH, trust_remote_code=True, load_in_8bit=True,
                                             torch_dtype=torch.bfloat16)
assert model.dtype == torch.bfloat16
"""
1.8B模型有24层、隐维度2048; 7B模型有32层、隐维度4096

QWenLMHeadModel(
  (transformer): QWenModel(
    (wte): Embedding(151936, 2048)
    (drop): Dropout(p=0.0, inplace=False)
    (rotary_emb): RotaryEmbedding()
    (h): ModuleList(
      (0-23): 24 x QWenBlock(
        (ln_1): RMSNorm()
        (attn): QWenAttention(
          (c_attn): Linear8bitLt(in_features=2048, out_features=6144, bias=True)
          (c_proj): Linear8bitLt(in_features=2048, out_features=2048, bias=False)
          (attn_dropout): Dropout(p=0.0, inplace=False)
        )
        (ln_2): RMSNorm()
        (mlp): QWenMLP(
          (w1): Linear8bitLt(in_features=2048, out_features=5504, bias=False)
          (w2): Linear8bitLt(in_features=2048, out_features=5504, bias=False)
          (c_proj): Linear8bitLt(in_features=5504, out_features=2048, bias=False)
        )
      )
    )
    (ln_f): RMSNorm()
  )
  (lm_head): Linear(in_features=2048, out_features=151936, bias=False)
)
"""

# 使用gradient_checkpointing+peft需要加这句补丁
# 量化模式下，可以用model=prepare_model_for_kbit_training(model)代替，顺便开了gradient_checkpointing（后面TrainingArguments可以删掉对应参数）
model.enable_input_require_grads()

# 定义LoRA配置
# 原本加上inference_mode=False（表示训练模式），这是默认值，省略
# 虽然是量化+lora，但并不是qlora（需要量化为nf4）
config = LoraConfig(
    # task_type=TaskType.CAUSAL_LM, # 不填写该参数也行
    target_modules=["c_attn", "c_proj", "w1", "w2"],  # 必须指定，Qwen官方的微调脚本也是写这几个
    # 原文说这三个参数比较通用，先留着
    r=8,
    lora_alpha=32,
    lora_dropout=0.1
)
"""
LoraConfig(peft_type=<PeftType.LORA: 'LORA'>, auto_mapping=None, base_model_name_or_path=None, revision=None, task_type=None, inference_mode=False, r=8, target_modules={'c_proj', 'c_attn', 'w2', 'w1'}, lora_alpha=32, lora_dropout=0.1, fan_in_fan_out=False, bias='none', use_rslora=False, modules_to_save=None, init_lora_weights=True, layers_to_transform=None, layers_pattern=None, rank_pattern={}, alpha_pattern={}, megatron_config=None, megatron_core='megatron.core', loftq_config={}, use_dora=False, layer_replication=None)
"""

# model加上LoRA插件
model = get_peft_model(model, config)
"""
PeftModel(
  (base_model): LoraModel(
    (model): QWenLMHeadModel(
      (transformer): QWenModel(
        (wte): Embedding(151936, 2048)
        (drop): Dropout(p=0.0, inplace=False)
        (rotary_emb): RotaryEmbedding()
        (h): ModuleList(
          (0-23): 24 x QWenBlock(
            (ln_1): RMSNorm()
            (attn): QWenAttention(
              (c_attn): lora.Linear8bitLt(
                (base_layer): Linear8bitLt(in_features=2048, out_features=6144, bias=True)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=2048, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=6144, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
              )
              (c_proj): lora.Linear8bitLt(
                (base_layer): Linear8bitLt(in_features=2048, out_features=2048, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=2048, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=2048, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
              )
              (attn_dropout): Dropout(p=0.0, inplace=False)
            )
            (ln_2): RMSNorm()
            (mlp): QWenMLP(
              (w1): lora.Linear8bitLt(
                (base_layer): Linear8bitLt(in_features=2048, out_features=5504, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=2048, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=5504, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
              )
              (w2): lora.Linear8bitLt(
                (base_layer): Linear8bitLt(in_features=2048, out_features=5504, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=2048, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=5504, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
              )
              (c_proj): lora.Linear8bitLt(
                (base_layer): Linear8bitLt(in_features=5504, out_features=2048, bias=False)
                (lora_dropout): ModuleDict(
                  (default): Dropout(p=0.1, inplace=False)
                )
                (lora_A): ModuleDict(
                  (default): Linear(in_features=5504, out_features=8, bias=False)
                )
                (lora_B): ModuleDict(
                  (default): Linear(in_features=8, out_features=2048, bias=False)
                )
                (lora_embedding_A): ParameterDict()
                (lora_embedding_B): ParameterDict()
              )
            )
          )
        )
        (ln_f): RMSNorm()
      )
      (lm_head): Linear(in_features=2048, out_features=151936, bias=False)
    )
  )
)
"""

"""
model.print_trainable_parameters()=
1.8B:  trainable params: 6,709,248 || all params: 1,843,537,920 || trainable%: 0.3639
7B:    trainable params: 17,891,328 || all params: 7,739,215,872 || trainable%: 0.2312
"""

# 定义训练配置
args = TrainingArguments(
    output_dir=f"./output/{CKPT_PATH}_new",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,  # 累计16个样本才迭代一次，1个epoch有1000/16=62步
    logging_steps=10,
    num_train_epochs=12,
    gradient_checkpointing=True,  # 开启激活重计算，时间换空间
    save_steps=186,  # 也就是正好3个epoch保存一次
    learning_rate=1e-4
)

# 定义trainer
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    # 用pad填充batch的三个字段
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer)  # 原文设置padding=True，是默认值，省略掉
)

# 训练前看看效果
model.eval().chat(tokenizer, "帕金森叠加综合征的辅助治疗有些什么?", history=None,
                  system="你是一个医学助手，需要回答用户关于医学的问题：")

# 正式训练
# 用什么loss是定义在QWenLMHeadModel里面的，通用做法
# 用CrossEntropyLoss实现的LM任务，同时该loss会忽略-100的label
trainer.train()  # 会自动调用model.train()
"""
TrainOutput(global_step=744, training_loss=1.3516380351076844, metrics={'train_runtime': 1978.4477, 'train_samples_per_second': 6.065, 'train_steps_per_second': 0.376, 'total_flos': 3.0783890072223744e+16, 'train_loss': 1.3516380351076844, 'epoch': 11.9})
"""

# 训练后看看效果（用回训练集测试效果当然好）
model.eval().chat(tokenizer, "帕金森叠加综合征的辅助治疗有些什么?", history=None,
                  system="你是一个医学助手，需要回答用户关于医学的问题：")
