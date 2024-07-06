# -*- coding: utf-8 -*-
# @file config.py
# @author zhangshilong
# @date 2024/7/6

import platform
from functools import partialmethod
from typing import Optional

import pandas as pd
from tqdm import tqdm
from transformers import set_seed

from utils import File


class ConfigMeta(type):
    def __init__(cls, name, bases, attr):
        super().__init__(name, bases, attr)
        tqdm.pandas()  # DataFrame添加progress_apply方法
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_colwidth", 500)  # 显示完整的文本

        if platform.system() == "Linux":
            [File.makedirs(v) for k, v in vars(cls).items() if k.endswith("_DIR")]


class Config(metaclass=ConfigMeta):
    QUESTION_NUM = 1000
    COMPANY_NUM = 80

    WORKSPACE_DIR = "/mnt/workspace"

    DATASET_DIR = File.join(WORKSPACE_DIR, "bs_challenge_financial_14b_dataset")
    DATABASE_PATH = File.join(DATASET_DIR, "dataset", "博金杯比赛数据.db")
    CID_PDF_DIR = File.join(DATASET_DIR, "pdf")
    CID_TXT_DIR = File.join(DATASET_DIR, "pdf_txt_file")
    QUESTION_PATH = File.join(DATASET_DIR, "question.json")

    REFERENCE_DIR = File.join(WORKSPACE_DIR, "reference")
    REF_COMPANY_PATH = File.join(REFERENCE_DIR, "AF0_pdf_to_company.csv")

    EXPERIMENT_DIR = File.join(WORKSPACE_DIR, "experiment")

    INTERMEDIATE_DIR = File.join(WORKSPACE_DIR, "intermediate")
    COMPANY_PATH = File.join(INTERMEDIATE_DIR, "A1_cid_to_company.csv")
    COMPANY_PDF_DIR = File.join(INTERMEDIATE_DIR, "pdf")
    COMPANY_TXT_DIR = File.join(INTERMEDIATE_DIR, "txt")

    @classmethod
    def company_pdf_path(cls, cid: Optional[str] = None, company: Optional[str] = None):
        assert bool(cid) ^ bool(company), "cid和company参数必须且只能填其中一个"
        if cid:
            return File.join(cls.CID_PDF_DIR, f"{cid}.PDF")
        else:
            return File.join(cls.COMPANY_PDF_DIR, f"{company}.pdf")

    @classmethod
    def company_txt_path(cls, cid: Optional[str] = None, company: Optional[str] = None):
        assert bool(cid) ^ bool(company), "cid和company参数必须且只能填其中一个"
        if cid:
            return File.join(cls.CID_TXT_DIR, f"{cid}.txt")
        else:
            return File.join(cls.COMPANY_TXT_DIR, f"{company}.txt")

    @classmethod
    def get_question_df(cls):
        question_df = pd.read_json(cls.QUESTION_PATH, lines=True)
        assert len(question_df) == cls.QUESTION_NUM
        return question_df

    @classmethod
    def get_ref_company_df(cls):
        ref_company_df = pd.read_csv(cls.REF_COMPANY_PATH)
        assert len(ref_company_df) == cls.COMPANY_NUM
        return ref_company_df

    @classmethod
    def set_seed(cls, seed: int = 1024) -> None:
        set_seed(seed)

    @classmethod
    def to_dict(cls):
        return {k: v for k, v in vars(cls).items() if
                not k.startswith("_") and not isinstance(v, (classmethod, partialmethod))}
