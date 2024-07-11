# -*- coding: utf-8 -*-
# @file test_LLM_classification_ability.py
# @author zhangshilong
# @date 2024/7/9
# 编写不同难度的prompt, 测试不同模型的回答能力

import pandas as pd

from ..tools.config import Config
from ..tools.constant import Category
from ..tools.constant import ModelName

# ============================================================================

# 调整模型
model_name = ModelName.QWEN_7B_CHAT
tokenizer = Config.get_tokenizer(model_name)
model = Config.get_model(model_name)

test_question_df = Config.get_classification_test_question_df()
company_df, companies = Config.get_company_df(return_companies=True)
tables = ["基金基本信息", "基金股票持仓明细", "基金债券持仓明细", "基金可转债持仓明细", "基金日行情表", "A股票日行情表",
          "港股票日行情表", "A股公司行业划分表", "基金规模变动表", "基金份额持有人结构"]

# ============================================================================


prompt1 = f"""请在以下公司中找出以“上海”开头的公司：

{"、".join(companies)}

"""
print(f"prompt:\n\n{prompt1}\n\n")

response1, _ = model.chat(tokenizer, prompt1, history=None)
print(f"回答:\n\n{response1}\n\n")

# ============================================================================


prompt2 = f"""任务描述：
请根据提供的资料回答最后的问题。


资料：
提供招股说明书的公司：{"、".join(companies)}



问题：提供招股说明书的公司中，有哪些在上海
回答："""
print(f"prompt:\n\n{prompt2}\n\n")

response2, _ = model.chat(tokenizer, prompt2, history=None)
print(f"回答:\n\n{response2}\n\n")


# ============================================================================


def apply_chat_func(chat_func, sample_num=None):
    if sample_num:
        sample_df = test_question_df.sample(sample_num, random_state=Config.SEED).sort_index()
    else:
        sample_df = test_question_df.copy()
    response_df = pd.concat([sample_df, sample_df.progress_apply(chat_func, axis=1)], axis=1)

    def category_func(row):
        response = row["回答"]

        # 精确匹配，更难
        # if response == "招股说明书":
        #     category = Category.TEXT
        # elif response == "基金股票数据库":
        #     category = Category.SQL
        # else:
        #     category = "Unknown"

        # 包含匹配，更易
        if "招股说明书" in response and "基金股票数据库" not in response:
            category = Category.TEXT
        elif "招股说明书" not in response and "基金股票数据库" in response:
            category = Category.SQL
        else:
            category = "Unknown"

        return pd.Series({"问题分类": category})

    category_df = pd.concat([response_df, response_df.progress_apply(category_func, axis=1)], axis=1)

    # 展示分类效果
    category_df["分类正确"] = category_df["问题标签"] == category_df["问题分类"]
    question_num = len(category_df)
    correct_num = category_df["分类正确"].sum()
    print(f"测试问题数： {question_num}")  # 110
    print(f"分类正确数：{correct_num}")
    print(f"分类正确率：{correct_num / question_num:.2%}")

    # 展示bad case
    print(f"\nbad case:")
    for _, row in category_df.query("分类正确 == False").iterrows():
        question = row["问题"]
        answer = row["回答"]
        label = row["问题标签"]
        category = row["问题分类"]
        print(f"{question=}")
        print(f"{answer=}")
        print(f"{label=}")
        print(f"{category=}")
        print()

    return category_df


# ============================================================================


prompt_template3 = f"""任务描述：
给定一个问题，你需要根据提供的资料猜测答案是在“招股说明书”还是“基金股票数据库”里，只需要进行分类，不需要回答问题。


资料：
提供招股说明书的公司：{"、".join(companies)}
基金股票数据库的表名：{"、".join(tables)}


请对以下问题进行分类：
问题：{{question}}
分类："""


def chat_func3(row):
    question = row["问题"]
    prompt = prompt_template3.format(question=question)
    response, _ = model.chat(tokenizer, prompt, history=None, system="你是一个问题分类器。")

    return pd.Series({"回答": response, "prompt": prompt})


category_df3 = apply_chat_func(chat_func3, 10)

# ============================================================================


prompt_template4 = f"""任务描述：
给定一个问题，你需要根据提供的资料猜测答案是在“招股说明书”还是“基金股票数据库”里，只需要进行分类，不需要回答问题。


资料：
提供招股说明书的公司：{"、".join(companies)}
基金股票数据库的表名：{"、".join(tables)}


例子：
问题：为什么广东银禧科技股份有限公司核心技术大部分为非专利技术？
分类：招股说明书

问题：读者出版传媒股份有限公司董事是谁？
分类：招股说明书

问题：在20201022，按照中信行业分类的行业划分标准，哪个一级行业的A股公司数量最多？
分类：基金股票数据库

问题：请帮我计算，代码为603937的股票，2020年一年持有的年化收益率有多少？百分数请保留两位小数。年化收益率定义为：（（有记录的一年的最终收盘价-有记录的一年的年初当天开盘价）/有记录的一年的当天开盘价）* 100%。
分类：基金股票数据库

问题：湖南南岭民用爆破器材股份有限公司主要业务是什么？
分类：招股说明书

问题：我想知道在20211231的季报里，中信保诚红利精选混合C投资的股票分别是哪些申万一级行业？
分类：基金股票数据库

问题：浙江开尔新材料股份有限公司成立时主要产品有什么？
分类：招股说明书

问题：帮我查一下广发瑞安精选股票A基金在20211222的资产净值和单位净值是多少?
分类：基金股票数据库


请对以下问题进行分类：
问题：{{question}}
分类："""


def chat_func4(row):
    question = row["问题"]
    prompt = prompt_template4.format(question=question)
    response, _ = model.chat(tokenizer, prompt, history=None, system="你是一个问题分类器。")

    return pd.Series({"回答": response, "prompt": prompt})


category_df4 = apply_chat_func(chat_func4, 10)

"""
结论：Qwen-14B-Chat-Int4表现最好，最适合做分类任务，下一步研究优化prompt

Qwen-14B-Chat-Int8效果和Qwen-14B-Chat-Int4差不多，但显存大、gptq没有用exllama kernel，性价比较低
Tongyi系列的效果较差，不遵循指令
详细的实验数据在notion
"""
