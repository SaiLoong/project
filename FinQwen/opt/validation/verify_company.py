# -*- coding: utf-8 -*-
# @file verify_company.py
# @author zhangshilong
# @date 2024/7/4

import json
import os
import re
import shutil
from collections import OrderedDict

import pandas as pd

COMPANY_NUM = 80
WORKSPACE_DIR = "/mnt/workspace"
DATASET_DIR = f"{WORKSPACE_DIR}/bs_challenge_financial_14b_dataset"
FILES_DIR = f"{WORKSPACE_DIR}/files"
VALIDATION_DIR = f"{WORKSPACE_DIR}/validation"

SRC_PDF_TXT_FILE_DIR = f"{DATASET_DIR}/pdf_txt_file"
DST_PDF_TXT_FILE_DIR = f"{VALIDATION_DIR}/pdf_txt_file"

# 原作者提取的映射表
company_df = pd.read_csv(f"{FILES_DIR}/AF0_pdf_to_company.csv")
assert len(company_df) == COMPANY_NUM

# 将文件用公司重命名，方便查阅
cid_to_company_dict = dict()  # cid = company_id
company_to_cid_dict = dict()
os.makedirs(DST_PDF_TXT_FILE_DIR, exist_ok=True)
for idx, row in company_df.iterrows():
    cid = row["csv文件名"].rsplit(".", 2)[0]
    company = row["公司名称"]
    cid_to_company_dict[cid] = company
    company_to_cid_dict[company] = cid

    src_filename = f"{cid}.txt"
    src_path = f"{SRC_PDF_TXT_FILE_DIR}/{src_filename}"
    dst_filename = f"{company}.txt"
    dst_path = f"{DST_PDF_TXT_FILE_DIR}/{dst_filename}"
    shutil.copy(src_path, dst_path)

with open(f"{VALIDATION_DIR}/cid_to_company.json", "w", encoding="utf-8") as f:
    json.dump(OrderedDict(sorted(cid_to_company_dict.items())), f)
with open(f"{VALIDATION_DIR}/company_to_cid.json", "w", encoding="utf-8") as f:
    json.dump(OrderedDict(sorted(company_to_cid_dict.items())), f)

# 根据文件内有无公司名分成若干类
entires = list()  # 公司名占一整行，肯定是
colons1 = dict()  # 公司名在冒号后面，冒号前有关键词
colons2 = dict()  # 公司名在冒号后面，冒号前没有关键词
founds = dict()  # 不是上面的情况，将包含公司名的行列出来再人工判断
unfounds = list()  # 没有提及公司名，一定有问题
companies = company_df["公司名称"].tolist()
for company in companies:
    with open(f"{DST_PDF_TXT_FILE_DIR}/{company}.txt", "r", encoding="utf-8") as f:
        is_entire = False
        all_evidences = list()
        colon1_evidences = list()
        colon2_evidences = list()

        for line in f:
            if company in line:
                all_evidences.append(line)
                if line == f"{company}\n":
                    is_entire = True
                    break
                # 冒号和公司名之间可以有一个空格
                elif m := re.fullmatch(f"(.*)： ?{company}\n", line):
                    left = m.group(1).replace(" ", "")
                    if re.search("企业名称|公司名称|中文名称|发行人", left):
                        colon1_evidences.append(line)
                    else:
                        colon2_evidences.append(line)

        if is_entire:
            entires.append(company)
        elif colon1_evidences:
            colons1[company] = colon1_evidences
        elif colon2_evidences:
            colons2[company] = colon2_evidences
        elif all_evidences:
            founds[company] = all_evidences
        else:
            unfounds.append(company)

# 分别是44、31、1、3、1个
# 经过人工复验，有3个错误。但是对比question，貌似将错就错也可以。。。
"""
founds:
旷达汽车织物集团股份有限公司 -> 江苏旷达汽车织物集团股份有限公司    question出现7次，只有1次（524）没有加江苏
山东海看网络科技有限公司    ->  海看网络科技（山东）股份有限公司   后者在文档中出现次数更多，但前者在question中出现11次

unfounds:
沈阳晶格自动化技术有限公司 -> 深圳麦格米特电气股份有限公司     question没有出现过
"""
