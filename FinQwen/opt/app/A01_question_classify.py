# -*- coding: utf-8 -*-
# @file A01_question_classify.py
# @author zhangshilong
# @date 2024/7/4

import csv

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import GenerationConfig

QUESTION_NUM = 1000
COMPANY_NUM = 80
WORKSPACE_DIR = "/mnt/workspace"
DATASET_DIR = f"{WORKSPACE_DIR}/bs_challenge_financial_14b_dataset"
INTERMEDIATE_DIR = f"{WORKSPACE_DIR}/intermediate"
FILES_DIR = f"{WORKSPACE_DIR}/files"

# 原版用14B-Chat，太大加载不了，只能换Int4版本
model_dir = f"{WORKSPACE_DIR}/Tongyi-Finance-14B-Chat-Int4"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

question_df = pd.read_csv(f"{INTERMEDIATE_DIR}/question_csv.csv")
assert len(question_df) == QUESTION_NUM

# TODO 有三个公司名不正确，但应该不影响回答，晚点处理
company_df = pd.read_csv(f"{FILES_DIR}/AF0_pdf_to_company.csv")
assert len(company_df) == COMPANY_NUM

company_list = company_df["公司名称"].tolist()

# 必须加device_map="cuda"，否则默认加载到cpu上。加载到gpu占用10594M显存
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, device_map="cuda").eval()
# 14B-Chat用bf16，但换成14B-Chat-Int4后用fp16
assert model.dtype == torch.float16
assert model.device.type == "cuda"

# 原本默认do_sample=True、top_p=0.8、没有设temperature。没有seed参数
model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True, do_sample=False)

# TODO prompt进一步优化方向
"""
"你是一个问题分类器"放到system prompt
XX的长度
22个shot会不会太多
"""
prompt_template = """你是一个问题分类器。对于每个提供给你的问题，你需要猜测答案是在该公司的招股说明书中还是在基金股票数据库里。以下是一些例子：

问题：在2019年的中期报告里，XX基金管理有限公司管理的基金中，有多少比例的基金是个人投资者持有的份额超过机构投资者？希望得到一个精确到两位小数的百分比。
回答：基金股票数据库

问题：XXXX股份有限公司变更设立时作为发起人的法人有哪些？
回答：该公司的招股说明书

问题：我想知道XXXXXX债券A基金在20200930的季报中，其可转债持仓占比最大的是哪个行业？用申万一级行业来统计。
回答：基金股票数据库

问题：XXXXXX股份有限公司2020年增资后的投后估值是多少？
回答：该公司的招股说明书

问题：根据XXXXXX股份有限公司招股意向书，全球率先整体用LED路灯替换传统路灯的案例是？
回答：该公司的招股说明书

问题：什么公司、在何时与XXXXXX股份有限公司发生了产品争议事项？产品争议事项是否已经解决？
回答：该公司的招股说明书

问题：请帮我查询下股票代码为XXXXXX的股票在2021年内最高日收盘价是多少？
回答：基金股票数据库

问题：XXXXXX股份有限公司的中标里程覆盖率为多少？
回答：该公司的招股说明书

问题：根据中国证监会颁布的《上市公司行业分类指导》的规定，XXXXXX有限公司所属的行业大类、中类、小类是什么？
回答：该公司的招股说明书

问题：请问XXXX年一季度有多少家基金是净申购?它们的净申购份额加起来是多少?请四舍五入保留小数点两位。
回答：基金股票数据库

问题：XXXXXX有限公司和合肥翰林是否按规定为员工缴纳了社会保险？
回答：该公司的招股说明书

问题：我想知道XXXXXX有限公司在2020年成立了多少只管理费率小于0.8%的基金？
回答：基金股票数据库

问题：根据《CRCC产品认证实施规则》，《铁路产品认证证书》有效期为多久？XXXXXX有限公司取得 《铁路产品认证证书》后，至少多久需要接受一次监督？
回答：该公司的招股说明书

问题：我想知道XXXXXX基金管理有限公司在2019年成立了多少只管理费率小于0.8%的基金？
回答：基金股票数据库

问题：请问XXXX年一季度有多少家基金是净申购?它们的净申购份额加起来是多少?请四舍五入保留小数点两位。
回答：基金股票数据库

问题：我想知道XXXXXX有限公司在2019年成立了多少只管理费率小于0.8%的基金？
回答：基金股票数据库”

问题：我想知道股票XXXXXX在申万行业分类下的二级行业是什么？用最新的数据。
回答：基金股票数据库

问题：请帮我查询下股票代码为XXXXXX的股票在2019年内最高日收盘价是多少？
回答：基金股票数据库

问题：股票XXXXXX在20200227日期中的收盘价是多少?（小数点保留3位）
回答：基金股票数据库

问题：截至2009年底，中海达、南方测绘合计占有国产品牌销售额的多大比例？
回答：该公司的招股说明书

问题：截止2005年12月31日，南岭化工厂的总资产和净资产分别是多少？
回答：该公司的招股说明书

问题：股票XXXXXX在20200227日期中的收盘价是多少?（小数点保留3位）
回答：基金股票数据库


根据上面提供的例子对以下问题进行分类：

问题：{question}
回答："""

# TODO 先取5-10条测试一下
# TODO ing 内部逻辑待优化，包括prompt、逐条chat、system指令、后处理、命名等
with open(f"{INTERMEDIATE_DIR}/A01_question_classify.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["问题id", "问题", "答案", "分类"])
    # tqdm拿不到迭代器的长度，只能主动赋值
    for index, row in tqdm(question_df.iterrows(), total=len(question_df)):
        question = row["问题"]
        prompt = prompt_template.format(question=question)
        response, history = model.chat(tokenizer, prompt, history=None)

        if "招股说明书" in response and "股票数据库" not in response:
            temp_class = "Text"
        elif "招股说明书" not in response and "股票数据库" in response:
            temp_class = "SQL"
            for company_name in company_list:
                if company_name in question:
                    temp_class = "Text"
        else:
            temp_class = "SQL"
            for company_name in company_list:
                if company_name in question:
                    temp_class = "Text"

        writer.writerow([row["问题id"], question, response, temp_class])

# TODO
"""
prompt模板是不是从question提取的？
"""
