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
# 显示完整的文本
pd.set_option("display.max_colwidth", 1000)

COMPANY_NUM = 80
SAMPLE_NUM = 100
WORKSPACE_DIR = "/mnt/workspace"
FILES_DIR = f"{WORKSPACE_DIR}/files"
VALIDATION_DIR = f"{WORKSPACE_DIR}/validation"

model_dir = f"{WORKSPACE_DIR}/Tongyi-Finance-14B-Chat-Int4"
# 如果用batch推理，需要左padding
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, padding_side="left")
tokenizer.pad_token_id = tokenizer.eod_id

model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, device_map="cuda").eval()
assert model.dtype == torch.float16
assert model.device.type == "cuda"

model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True, do_sample=False)
# 添加batch方法
model.batch = MethodType(batch, model)

# 人工标注数据
test_df = pd.read_json(f"{VALIDATION_DIR}/question_test.json")
assert len(test_df) == SAMPLE_NUM

# 获取公司名
company_df = pd.read_csv(f"{FILES_DIR}/AF0_pdf_to_company.csv")
assert len(company_df) == COMPANY_NUM
company_list = company_df["公司名称"].tolist()

prompt_template_ref_v5 = """对于每个提供给你的问题，你需要猜测答案是在该公司的招股说明书中还是在基金股票数据库里。以下是一些例子：

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


def chat_func_ref_v5(row):
    question = row["问题"]
    prompt = prompt_template_ref_v5.format(question=question)
    response, history = model.chat(tokenizer, prompt, history=None, system="你是一个问题分类器。")

    return pd.Series({"回答": response})


chat_func = chat_func_ref_v5
response_df = pd.concat([test_df, test_df.progress_apply(chat_func, axis=1)], axis=1)


def category_func_ref_v2(row):
    question = row["问题"]
    response = row["回答"]

    for company_name in company_list:
        if company_name in question:
            category = "Text"
            break
    else:
        if "招股说明书" in response and "股票数据库" not in response:
            category = "Text"
        else:
            category = "SQL"

    # 纯靠LLM
    if "招股说明书" in response and "股票数据库" not in response:
        raw_category = "Text"
    elif "招股说明书" not in response and "股票数据库" in response:
        raw_category = "SQL"
    else:
        raw_category = "Unknown"

    # 完全靠规则，不依赖LLM
    for company_name in company_list:
        if company_name in question:
            rule_category = "Text"
            break
    else:
        rule_category = "SQL"

    return pd.Series({"分类": category, "原始分类": raw_category, "规则分类": rule_category})


category_func = category_func_ref_v2
answer_df = pd.concat([response_df, response_df.progress_apply(category_func, axis=1)], axis=1)

answer_df["分类正确"] = answer_df["标签"] == answer_df["分类"]
print(f'分类正确数：{answer_df["分类正确"].sum()}')
answer_df.query("分类正确 == False")

answer_df["原始分类正确"] = answer_df["标签"] == answer_df["原始分类"]
print(f'原始分类正确数：{answer_df["原始分类正确"].sum()}')
answer_df.query("原始分类正确 == False")

answer_df["规则分类正确"] = answer_df["标签"] == answer_df["规则分类"]
print(f'规则分类正确数：{answer_df["规则分类正确"].sum()}')
answer_df.query("规则分类正确 == False")


# TODO 实验结果
"""
我的直接处理 94

ref经过后处理 99
    逐个chat，耗时 2:05，显存15116M
        原始正确数81，后处理正确数99
        有43个问题包含公司名，15个本来预测错误，靠着公司名纠正了
        
        id=511，正确答案是SQL，回答成Text
        "2021年年度报告里，光大保德信基金管理有限公司管理的基金中，机构投资者持有份额比个人投资者多的基金有多少只?"
    
    batch_size=4，耗时 2:04，显存22502M（上限是22731M，快爆了）
        速度没有提升可能是因为GPU利用率已经达到100%了，而1.8B-chat则没有，
        id=879，正确答案是SQL，回答成Text
        "生态环境建设行业的上游是什么行业？"

换成1.8B-chat:
    逐个chat, 耗时5:18, GPU利用率46%左右，显存6588M
        原始正确数49，后处理正确数99。将样例大段大段地拷贝下来
        
    batch_size=16，耗时0:40, GPU利用率94%左右，显存20118M
        原始正确数31，后处理正确数99。

"""
# TODO 从ref的方案逐步修改，看是哪里导致准确率下降
"""
TODO
1. 检查1000问题中有多少个包含公司名，以及公司名覆盖度是否广
2. 再标100条（与之前100条去重），看效果是否还是那么好


既然可以靠公司名纠正，是不是也可以靠基金名纠正SQL
"""
