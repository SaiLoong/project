# -*- coding: utf-8 -*-
# @file classify_test_question_by_LLM.py
# @author zhangshilong
# @date 2024/7/7

import pandas as pd

from ..tools.config import Config
from ..tools.constant import Category

tokenizer = Config.get_tokenizer()
model = Config.get_model()

test_question_df = Config.get_test_question_df()
company_df, companies = Config.get_company_df(return_companies=True)

# 从原作者的prompt_template逐渐修改，优化到可读性、效果最优，迭代过程记录在classification_prompt_storage.py
prompt_template_v5 = """对于每个提供给你的问题，你需要猜测答案是在该公司的招股说明书中还是在基金股票数据库里。以下是一些例子：

问题：在2019年的中期报告里，XX基金管理有限公司管理的基金中，有多少比例的基金是个人投资者持有的份额超过机构投资者？希望得到一个精确到两位小数的百分比。
回答：基金股票数据库

问题：XX股份有限公司变更设立时作为发起人的法人有哪些？
回答：该公司的招股说明书

问题：我想知道XX债券A基金在20200930的季报中，其可转债持仓占比最大的是哪个行业？用申万一级行业来统计。
回答：基金股票数据库

问题：XX股份有限公司2020年增资后的投后估值是多少？
回答：该公司的招股说明书

问题：根据XX股份有限公司招股意向书，全球率先整体用LED路灯替换传统路灯的案例是？
回答：该公司的招股说明书

问题：什么公司、在何时与XX股份有限公司发生了产品争议事项？产品争议事项是否已经解决？
回答：该公司的招股说明书

问题：请帮我查询下股票代码为XX的股票在2021年内最高日收盘价是多少？
回答：基金股票数据库

问题：XX股份有限公司的中标里程覆盖率为多少？
回答：该公司的招股说明书

问题：根据中国证监会颁布的《上市公司行业分类指导》的规定，XX有限公司所属的行业大类、中类、小类是什么？
回答：该公司的招股说明书

问题：请问XX年一季度有多少家基金是净申购?它们的净申购份额加起来是多少?请四舍五入保留小数点两位。
回答：基金股票数据库

问题：XX有限公司和合肥翰林是否按规定为员工缴纳了社会保险？
回答：该公司的招股说明书

问题：我想知道XX有限公司在2020年成立了多少只管理费率小于0.8%的基金？
回答：基金股票数据库

问题：根据《CRCC产品认证实施规则》，《铁路产品认证证书》有效期为多久？XX有限公司取得 《铁路产品认证证书》后，至少多久需要接受一次监督？
回答：该公司的招股说明书

问题：我想知道XX基金管理有限公司在2019年成立了多少只管理费率小于0.8%的基金？
回答：基金股票数据库

问题：请问XX年一季度有多少家基金是净申购?它们的净申购份额加起来是多少?请四舍五入保留小数点两位。
回答：基金股票数据库

问题：我想知道XX有限公司在2019年成立了多少只管理费率小于0.8%的基金？
回答：基金股票数据库

问题：我想知道股票XX在申万行业分类下的二级行业是什么？用最新的数据。
回答：基金股票数据库

问题：请帮我查询下股票代码为XX的股票在2019年内最高日收盘价是多少？
回答：基金股票数据库

问题：股票XX在20200227日期中的收盘价是多少?（小数点保留3位）
回答：基金股票数据库

问题：截至2009年底，中海达、南方测绘合计占有国产品牌销售额的多大比例？
回答：该公司的招股说明书

问题：截止2005年12月31日，南岭化工厂的总资产和净资产分别是多少？
回答：该公司的招股说明书

问题：股票XX在20200227日期中的收盘价是多少?（小数点保留3位）
回答：基金股票数据库

根据上面提供的例子对以下问题进行分类。
问题：{question}
回答："""


def chat_func_v5(row):
    question = row["问题"]
    prompt = prompt_template_v5.format(question=question)
    response, _ = model.chat(tokenizer, prompt, history=None, system="你是一个问题分类器。")

    return pd.Series({"回答": response})


chat_func = chat_func_v5
response_df = pd.concat([test_question_df, test_question_df.progress_apply(chat_func, axis=1)], axis=1)


def category_func(row):
    response = row["回答"]

    if "招股说明书" in response and "股票数据库" not in response:
        category = Category.TEXT
    elif "招股说明书" not in response and "股票数据库" in response:
        category = Category.SQL
    else:
        category = "Unknown"

    return pd.Series({"分类": category})


category_df = pd.concat([response_df, response_df.progress_apply(category_func, axis=1)], axis=1)

# 展示分类效果
category_df["分类正确"] = category_df["标签"] == category_df["分类"]
print(f'分类正确数：{category_df["分类正确"].sum()}')
category_df.query("分类正确 == False")

# 保存下来，不要浪费
category_df.to_csv(f"{Config.EXPERIMENT_DIR}/question_category_by_LLM_v5.csv", index=False)

"""
结论：V0原版纯LLM判断正确率81%，V5版本升至96%，后来发现规则直接100%，放弃这条路了

chat耗时大约在1.5min-2min左右，尝试过batch（batch_size=4接近爆显存），但速度没有丝毫提升，可能是因为本来GPU利用率已达到100%
如果换成1.8B-Chat, chat接口耗时5:18、GPU利用率46%，batch接口耗时0:40、GPU利用率100%，有提升（用时长是因为废话连篇，正确率只有49%/31%）
"""
