# -*- coding: utf-8 -*-
# @file classify_test_question_by_LLM.py
# @author zhangshilong
# @date 2024/7/7

import json
from string import Template

import pandas as pd

from ..tools.config import Config
from ..tools.constant import Category
from ..tools.constant import ModelName

# 根据test_LLM_classification_ability.py的结论，Qwen-14B-Chat-Int4做分类任务效果最好
model_name = ModelName.QWEN_14B_CHAT_INT4
tokenizer = Config.get_tokenizer(model_name)
model = Config.get_model(model_name)

test_question_df = Config.get_test_question_df()

company_df, companies = Config.get_company_df(return_companies=True)

# TODO 先手动填写，不查表
tables = ["基金基本信息", "基金股票持仓明细", "基金债券持仓明细", "基金可转债持仓明细", "基金日行情表", "A股票日行情表",
          "港股票日行情表", "A股公司行业划分表", "基金规模变动表", "基金份额持有人结构"]

# TODO 看到时有没更好的格式写法
company_str = "、".join(companies)
table_str = "、".join(tables)

# TODO 这些例子是test表以外的原题！！！
prompt_template_v15 = Template("""任务描述：
你需要根据资料将问题分类为"Text"或者"SQL"，以json格式输出。不要回答问题。


资料：
提供招股说明书的公司：$company_str
基金股票数据库的表名：$table_str


下面是一些例子：
问题：为什么广东银禧科技股份有限公司核心技术大部分为非专利技术？
输出：{"问题分类": "Text", "公司名称": "广东银禧科技股份有限公司"}

问题：读者出版传媒股份有限公司董事是谁？
输出：{"问题分类": "Text", "公司名称": "读者出版传媒股份有限公司"}

问题：在20201022，按照中信行业分类的行业划分标准，哪个一级行业的A股公司数量最多？
输出：{"问题分类": "SQL"}

问题：请帮我计算，代码为603937的股票，2020年一年持有的年化收益率有多少？百分数请保留两位小数。年化收益率定义为：（（有记录的一年的最终收盘价-有记录的一年的年初当天开盘价）/有记录的一年的当天开盘价）* 100%。
输出：{"问题分类": "SQL"}

问题：湖南南岭民用爆破器材股份有限公司主要业务是什么？
输出：{"问题分类": "Text", "公司名称": "湖南南岭民用爆破器材股份有限公司"}

问题：我想知道在20211231的季报里，中信保诚红利精选混合C投资的股票分别是哪些申万一级行业？
输出：{"问题分类": "SQL"}

问题：浙江开尔新材料股份有限公司成立时主要产品有什么？
输出：{"问题分类": "Text", "公司名称": "浙江开尔新材料股份有限公司"}

问题：帮我查一下广发瑞安精选股票A基金在20211222的资产净值和单位净值是多少?
输出：{"问题分类": "SQL"}


请参考上面例子对以下问题进行分类，并严格按照例子的格式输出json：
问题：$question
输出：""")


def chat_func_v15(row):
    question = row["问题"]
    prompt = prompt_template_v15.substitute(question=question, company_str=company_str, table_str=table_str)
    response, _ = model.chat(tokenizer, prompt, history=None, system="你是一个问题分类器。")

    return pd.Series({"回答": response, "prompt": prompt})


chat_func = chat_func_v15
df = test_question_df.sample(10, random_state=Config.SEED).sort_index()
# df = test_question_df
response_df = pd.concat([df, df.progress_apply(chat_func, axis=1)], axis=1)


# batch代码
# batch_size = 2
# responses = list()
# for start in tqdm(range(0, len(test_question_df), batch_size)):
#     chunk = test_question_df.iloc[start:start + batch_size]
#     questions = chunk["问题"].tolist()
#     prompts = [prompt_template_v5.format(question=question) for question in questions]
#     responses.extend(model.batch(tokenizer, prompts, system="你是一个问题分类器。"))
#
# response_df = pd.concat([test_question_df, pd.Series(responses, name="回答")], axis=1)


def category_func_v2(row):
    raw_response = row["回答"]

    category = None
    company = None
    try:
        response = json.loads(raw_response)
        category = response["问题分类"]
        if category == Category.TEXT:
            company = response["公司名称"]
    except Exception:
        pass

    return pd.Series({"问题分类": category, "公司名称识别": company})


category_func = category_func_v2
category_df = pd.concat([response_df, response_df.progress_apply(category_func, axis=1)], axis=1)

# 展示分类效果
category_df["分类正确"] = category_df["问题标签"] == category_df["问题分类"]
question_num = len(category_df)
correct_num = category_df["分类正确"].sum()
print(f"测试问题数： {question_num}")  # 110
print(f"分类正确数：{correct_num}")
print(f"分类正确率：{correct_num / question_num:.2%}")
# 展示bad case
category_df.query("分类正确 == False")

# 展示公司名称识别效果
text_df = category_df.query(f"问题标签 == '{Category.TEXT}'").copy()
text_df["公司名称识别正确"] = text_df["公司名称"] == text_df["公司名称识别"]
text_num = len(text_df)
company_correct_num = text_df["公司名称识别正确"].sum()
print(f"测试Text问题数： {text_num}")
print(f"公司名称识别正确数：{company_correct_num}")
print(f"公司名称识别正确率：{company_correct_num / text_num:.2%}")
# 展示bad case
text_df.query("公司名称识别正确 == False")

# 保存下来，不要浪费
category_df.to_csv(f"{Config.EXPERIMENT_OUTPUT_DIR}/question_category_by_LLM_v15.csv", index=False)

"""
结论：


chat耗时大约在1.5min-2min左右，尝试过batch（batch_size=4接近爆显存），但速度没有丝毫提升，可能是因为本来GPU利用率已达到100%
如果换成1.8B-Chat, chat接口耗时5:18、GPU利用率46%，batch接口耗时0:40、GPU利用率100%，有提升（用时长是因为废话连篇，正确率只有49%/31%）
"""
