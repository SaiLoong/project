# -*- coding: utf-8 -*-
# @file A1_cid_to_company.py
# @author zhangshilong
# @date 2024/7/6
# 构建包含 公司id 与 公司名称 的csv表格
# 基于原作者整理好的映射表修正，也可以通过正则匹配文档的方式从头构造

import pandas as pd

from ..tools.config import Config
from ..tools.utils import File

# 加载原映射表
ref_company_df = Config.get_ref_company_df()


def func(row):
    cid = row["csv文件名"].rsplit(".PDF.csv", 1)[0]
    company = row["公司名称"]
    return pd.Series({"公司id": cid, "公司名称": company})


company_df = ref_company_df.progress_apply(func, axis=1)

# 根据experiment的verify_company_name.py和verify_company_in_question.py分析结论，进行修正
company_df.replace({"公司名称": "沈阳晶格自动化技术有限公司"}, "深圳麦格米特电气股份有限公司", inplace=True)
assert company_df.iloc[33]["公司名称"] == "深圳麦格米特电气股份有限公司"

# 保存修正后的映射表
company_df.to_csv(Config.COMPANY_PATH, index=False)


# 将txt文件和pdf文件拷贝成中文文件名，方遍浏览
def func(row):
    cid = row["公司id"]
    company = row["公司名称"]

    src_path = Config.company_pdf_path(cid=cid)
    dst_path = Config.company_pdf_path(company=company)
    File.copy(src_path, dst_path, cover=True)

    src_path = Config.company_txt_path(cid=cid)
    dst_path = Config.company_txt_path(company=company)
    File.copy(src_path, dst_path, cover=True)


company_df.progress_apply(func, axis=1)
