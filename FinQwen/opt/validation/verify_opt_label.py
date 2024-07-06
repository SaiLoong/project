# -*- coding: utf-8 -*-
# @file verify_opt_label.py
# @author zhangshilong
# @date 2024/7/5

from types import MethodType

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import GenerationConfig

from qwen_generation_utils import batch

# DataFrame添加progress_apply方法
tqdm.pandas()

COMPANY_NUM = 80
SAMPLE_NUM = 100
WORKSPACE_DIR = "/mnt/workspace"
FILES_DIR = f"{WORKSPACE_DIR}/files"
VALIDATION_DIR = f"{WORKSPACE_DIR}/validation"

# 加载人工标注数据
test_df = pd.read_json(f"{VALIDATION_DIR}/question_test.json")
assert len(test_df) == SAMPLE_NUM

model_dir = f"{WORKSPACE_DIR}/Tongyi-Finance-14B-Chat-Int4"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, device_map="cuda").eval()
assert model.dtype == torch.float16
assert model.device.type == "cuda"

model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True, do_sample=False)
# 添加batch方法
model.batch = MethodType(batch, model)
# ==============================================================================================

company_df = pd.read_csv(f"{FILES_DIR}/AF0_pdf_to_company.csv")
assert len(company_df) == COMPANY_NUM
company_list = company_df["公司名称"].tolist()

# TODO prompt进一步优化方向
"""
22个shot会不会太多，更少的shot、zero-shot效果如何
逐条chat
后处理
"""
prompt_template_opt = """对于每个提供给你的问题，你需要猜测答案是在该公司的招股说明书中还是在基金股票数据库里。
如果你比较确定，就回答“基金股票数据库”或“该公司的招股说明书”；如果不太确定，就回答“不确定”。除此之外不要给出其它回答，也不需要给出解释
以下是一些例子：

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


请根据上面提供的例子对以下问题进行分类：

问题：{question}
回答："""


def chat_func_opt(row):
    question = row["问题"]
    response = None

    for company in company_list:
        if company in question:
            category = "Text"
            break
    else:
        prompt = prompt_template_opt.format(question=question)
        response, history = model.chat(tokenizer, prompt, history=None,
                                       system="你是一个擅长分类金融问题的助手"
                                       )
        if response == "该公司的招股说明书":
            category = "Text"
        elif response == "基金股票数据库":
            category = "SQL"
        elif response == "不确定":
            category = "Unknown"
        else:
            category = "Undefined"

        # if "招股说明书" in response and "股票数据库" not in response:
        #     category = "Text"
        # elif "招股说明书" not in response and "股票数据库" in response:
        #     category = "SQL"
        # elif "招股说明书" in response and "股票数据库" in response:
        #     category = "Both"
        # else:
        #     category = "Unknown"

    return pd.Series({"回答": response, "分类": category})


prompt_template_ref = """
    你是一个问题分类器。对于每个提供给你的问题，你需要猜测答案是在该公司的招股说明书中还是在基金股票数据库里。以下是一些例子：

    问题：“在2019年的中期报告里，XX基金管理有限公司管理的基金中，有多少比例的基金是个人投资者持有的份额超过机构投资者？希望得到一个精确到两位小数的百分比。”
    回答：“基金股票数据库”

    问题：“XXXX股份有限公司变更设立时作为发起人的法人有哪些？”
    回答：“该公司的招股说明书”

    问题：“我想知道XXXXXX债券A基金在20200930的季报中，其可转债持仓占比最大的是哪个行业？用申万一级行业来统计。”
    回答：“基金股票数据库”

    问题：“XXXXXX股份有限公司2020年增资后的投后估值是多少？”
    回答：“该公司的招股说明书”

    问题：“根据XXXXXX股份有限公司招股意向书，全球率先整体用LED路灯替换传统路灯的案例是？”
    回答：“该公司的招股说明书”

    问题：“什么公司、在何时与XXXXXX股份有限公司发生了产品争议事项？产品争议事项是否已经解决？”
    回答：“该公司的招股说明书”

    问题：“请帮我查询下股票代码为XXXXXX的股票在2021年内最高日收盘价是多少？”
    回答：“基金股票数据库”

    问题：“XXXXXX股份有限公司的中标里程覆盖率为多少？”
    回答：“该公司的招股说明书”

    问题：“根据中国证监会颁布的《上市公司行业分类指导》的规定，XXXXXX有限公司所属的行业大类、中类、小类是什么？”
    回答：“该公司的招股说明书”

    问题：“请问XXXX年一季度有多少家基金是净申购?它们的净申购份额加起来是多少?请四舍五入保留小数点两位。”
    回答：“基金股票数据库”

    问题：“XXXXXX有限公司和合肥翰林是否按规定为员工缴纳了社会保险？”
    回答：“该公司的招股说明书”

    问题：“我想知道XXXXXX有限公司在2020年成立了多少只管理费率小于0.8%的基金？”
    回答：“基金股票数据库”

    问题：“根据《CRCC产品认证实施规则》，《铁路产品认证证书》有效期为多久？XXXXXX有限公司取得 《铁路产品认证证书》后，至少多久需要接受一次监督？”
    回答：“该公司的招股说明书”

    问题：“我想知道XXXXXX基金管理有限公司在2019年成立了多少只管理费率小于0.8%的基金？”
    回答：“基金股票数据库”

    问题：“请问XXXX年一季度有多少家基金是净申购?它们的净申购份额加起来是多少?请四舍五入保留小数点两位。”
    回答：“基金股票数据库”

    问题：“我想知道XXXXXX有限公司在2019年成立了多少只管理费率小于0.8%的基金？”
    回答：“基金股票数据库”

    问题：“我想知道股票XXXXXX在申万行业分类下的二级行业是什么？用最新的数据。”
    回答：“基金股票数据库”

    问题：“请帮我查询下股票代码为XXXXXX的股票在2019年内最高日收盘价是多少？”
    回答：“基金股票数据库”

    问题：“股票XXXXXX在20200227日期中的收盘价是多少?（小数点保留3位）”
    回答：“基金股票数据库”

    问题：“截至2009年底，中海达、南方测绘合计占有国产品牌销售额的多大比例？”
    回答：“该公司的招股说明书”

    问题：“截止2005年12月31日，南岭化工厂的总资产和净资产分别是多少？”
    回答：“该公司的招股说明书”

    问题：“股票XXXXXX在20200227日期中的收盘价是多少?（小数点保留3位）”
    回答：“基金股票数据库”

    根据上面提供的例子对以下问题进行分类。
    问题：“
    """


def chat_func_ref(row):
    question = row["问题"]
    prompt = prompt_template_ref + question + """？"""
    response, history = model.chat(tokenizer, prompt, history=None)

    return pd.Series({"回答": response})


chat_func = chat_func_ref
response_df = pd.concat([test_df, test_df.progress_apply(chat_func, axis=1)], axis=1)


def category_func_ref(row):
    question = row["问题"]
    response = row["回答"]

    if "招股说明书" in response and "股票数据库" not in response:
        category = "Text"
        raw_category = "Text"
    elif "招股说明书" not in response and "股票数据库" in response:
        category = "SQL"
        raw_category = "SQL"
        for company_name in company_list:
            if company_name in question:
                category = "Text"
    else:
        category = "SQL"
        raw_category = "Unknown"
        for company_name in company_list:
            if company_name in question:
                category = "Text"

    return pd.Series({"分类": category, "原始分类": raw_category})


category_func = category_func_ref
answer_df = pd.concat([response_df, response_df.progress_apply(category_func, axis=1)], axis=1)

answer_df["分类正确"] = answer_df["标签"] == answer_df["分类"]
answer_df["分类正确"].sum()
answer_df.query("分类正确 == False")

answer_df["原始分类正确"] = answer_df["标签"] == answer_df["原始分类"]
answer_df["原始分类正确"].sum()
answer_df.query("原始分类正确 == False")

"""
我的直接处理 94

ref经过后处理 99
    id=511，正确答案是SQL，回答成Text
    "2021年年度报告里，光大保德信基金管理有限公司管理的基金中，机构投资者持有份额比个人投资者多的基金有多少只?"
    
    逐个chat，耗时 2:05

"""
