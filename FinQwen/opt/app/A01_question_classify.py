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

# TODO 原版用14B-Chat，太大加载不了，只能换Int4版本（或者14B-Chat+load_in_8_bit）
model_dir = f"{WORKSPACE_DIR}/Tongyi-Finance-14B-Chat-Int4"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)

question_df = pd.read_csv(f"{INTERMEDIATE_DIR}/question_csv.csv")
assert len(question_df) == QUESTION_NUM

# TODO
"""
readme解释：
使用正则表达式抽取公司名称。将举办方提供的每个text格式招股书文件分为1000字每段，
相邻两段具有200字重叠的片段，使用Qwen大模型的tokenizer进行词频统计备用

应该和Qwen Demo的做法差不多，80间公司不算多，人工检验+校准bad case不难
最后的词频应该放在AD_normalized_ot.csv，后面会用到
14B-Chat和14B-Chat-Int4的词汇表是一致的，可放心使用
"""
company_df = pd.read_csv(f"{FILES_DIR}/AF0_pdf_to_company.csv")
assert len(company_df) == COMPANY_NUM

company_list = company_df["公司名称"].tolist()

# TODO 原本加载14B-Chat用bf16，但换成14B-Chat-Int4后用fp16
# 必须加device_map="cuda"，否则默认加载到cpu上。加载到gpu占用10594M显存
model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, device_map="cuda").eval()
assert model.dtype == torch.float16
assert model.device.type == "cuda"

# TODO do_sample=False的话，temperature和top_p就没有意义了。不过temperature设这么小，貌似真的想贪心
# 原本默认do_sample=True、top_p=0.8、没有设temperature
model.generation_config = GenerationConfig.from_pretrained(model_dir, trust_remote_code=True,
                                                           temperature=0.0000001,
                                                           top_p=1,
                                                           do_sample=False,
                                                           seed=1234)

# TODO prompt问题挺多：模板行首有空格、尾行换行，后面拼接问题时格式没对齐、没有后引号
prompt = """
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

# TODO ing 内部逻辑待优化，包括prompt、逐条chat、system指令、后处理、命名等
with open(f"{INTERMEDIATE_DIR}/A01_question_classify.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["问题id", "问题", "答案", "分类"])
    # tqdm拿不到迭代器的长度，只能主动赋值
    for index, row in tqdm(question_df.iterrows(), total=len(question_df)):
        question = row["问题"]
        prompt1 = prompt + question + """？"""
        response_new, history_new = model.chat(tokenizer, prompt1, history=None)

        if "招股说明书" in response_new and "股票数据库" not in response_new:
            temp_class = "Text"
        elif "招股说明书" not in response_new and "股票数据库" in response_new:
            temp_class = "SQL"
            for company_name in company_list:
                if company_name in question:
                    temp_class = "Text"
        else:
            temp_class = "SQL"
            for company_name in company_list:
                if company_name in question:
                    temp_class = "Text"

        # TODO 这应该是复赛的数据。但写错了变量名temp_calss，没发挥作用？？？？
        if index in [166, 174]:
            temp_calss = "Text"

        writer.writerow([row["问题id"], question, response_new, temp_class])

# TODO 待探寻的点
"""
为什么模型不自动加载cuda
加载模型时冒出的”CUDA extension not installed.“有无问题
do_sample=false会怎么走
temperature默认多少
"""
