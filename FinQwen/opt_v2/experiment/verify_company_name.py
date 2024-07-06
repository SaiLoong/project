# -*- coding: utf-8 -*-
# @file verify_company_name.py
# @author zhangshilong
# @date 2024/7/6

import re

import pandas as pd

from ..tools.config import Config
from ..tools.utils import File

Config.set_seed()

# 原作者整理好的映射表
ref_company_df = Config.get_ref_company_df()

# 将文件用公司重命名，方便查阅
dst_dir = f"{Config.EXPERIMENT_DIR}/company_txt"
File.makedirs(dst_dir)


def func(row):
    cid = row["csv文件名"].rsplit(".PDF.csv", 1)[0]
    company = row["公司名称"]

    src_path = Config.company_txt_path(cid=cid)
    dst_path = f"{dst_dir}/{company}.txt"
    File.copy(src_path, dst_path, cover=True)

    return pd.Series({"公司id": cid, "公司名称": company})


company_df = ref_company_df.progress_apply(func, axis=1)

# 根据文件内有无公司名分成若干类
entires = list()  # 公司名占一整行，肯定正确
colons1 = dict()  # 公司名在冒号后面，冒号前有关键词，也正确
colons2 = dict()  # 公司名在冒号后面，冒号前没有关键词，需要人工复核
founds = dict()  # 不是上面的情况，需要人工复核
unfounds = list()  # 没有提及公司名，一定是错的
companies = company_df["公司名称"].tolist()
for company in companies:
    with open(f"{dst_dir}/{company}.txt", "r", encoding="utf-8") as f:
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

print(f"{len(entires)=} {len(colons1)=} {len(colons2)=} {len(founds)=} {len(unfounds)=}")
# 分别是44、31、1、3、1个


"""
结论：经过人工复核，有3个错误。founds的两个对问题检索有利，维持不变；unfounds的需要修正

founds:
旷达汽车织物集团股份有限公司 -> 江苏旷达汽车织物集团股份有限公司    question出现7次，只有1次（524）没有加江苏
山东海看网络科技有限公司    ->  海看网络科技（山东）股份有限公司   后者在文档中出现次数更多，但前者在question中出现11次（前者1次也没有出现过）

unfounds:
沈阳晶格自动化技术有限公司 -> 深圳麦格米特电气股份有限公司     question没有出现过
"""
