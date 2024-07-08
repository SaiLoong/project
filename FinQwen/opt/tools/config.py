# -*- coding: utf-8 -*-
# @file config.py
# @author zhangshilong
# @date 2024/7/6

import platform
from functools import partialmethod
from types import MethodType
from typing import Dict
from typing import Optional

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import GenerationConfig
from transformers import set_seed

from constant import ModelMode
from constant import ModelName
from qwen_utils import batch
from qwen_utils import batch_decode_each
from qwen_utils import cut
from qwen_utils import decode_each
from utils import File


class ConfigMeta(type):
    def __init__(cls, name, bases, attr):
        super().__init__(name, bases, attr)
        tqdm.pandas()  # DataFrame添加progress_apply方法
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_colwidth", 500)  # 显示完整的文本

        cls.set_seed()
        if platform.system() == "Linux":
            [File.makedirs(v) for k, v in vars(cls).items() if k.endswith("_DIR")]


class Config(metaclass=ConfigMeta):
    SEED = 1024
    QUESTION_NUM = 1000
    TEST_QUESTION_NUM = 100
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
    EXPERIMENT_OUTPUT_DIR = File.join(EXPERIMENT_DIR, "output")
    EXPERIMENT_REFERENCE_DIR = File.join(EXPERIMENT_DIR, "reference")

    INTERMEDIATE_DIR = File.join(WORKSPACE_DIR, "intermediate")
    COMPANY_PATH = File.join(INTERMEDIATE_DIR, "A1_cid_to_company.csv")
    COMPANY_PDF_DIR = File.join(INTERMEDIATE_DIR, "pdf")
    COMPANY_TXT_DIR = File.join(INTERMEDIATE_DIR, "txt")
    TEST_QUESTION_PATH = File.join(INTERMEDIATE_DIR, "test_question.csv")
    QUESTION_CATEGORY_PATH = File.join(INTERMEDIATE_DIR, "A2_question_category.csv")

    MODEL_NAME = ModelName.TONGYI_FINANCE_14B_CHAT_INT4

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
    def model_dir(cls, model_name: Optional[str] = None):
        model_name = model_name or cls.MODEL_NAME
        assert model_name in ModelName.values()
        return File.join(cls.WORKSPACE_DIR, model_name)

    @classmethod
    def get_question_df(cls, rename: bool = True):
        question_df = pd.read_json(cls.QUESTION_PATH, lines=True)
        assert len(question_df) == cls.QUESTION_NUM
        if rename:
            question_df.rename(columns={"id": "问题id", "question": "问题"}, inplace=True)
        return question_df

    @classmethod
    def get_ref_company_df(cls):
        ref_company_df = pd.read_csv(cls.REF_COMPANY_PATH)
        assert len(ref_company_df) == cls.COMPANY_NUM
        return ref_company_df

    @classmethod
    def get_company_df(cls, return_companies: bool = False):
        company_df = pd.read_csv(cls.COMPANY_PATH)
        assert len(company_df) == cls.COMPANY_NUM
        if return_companies:
            return company_df, company_df["公司名称"].tolist()
        else:
            return company_df

    @classmethod
    def get_test_question_df(cls):
        test_question_df = pd.read_csv(cls.TEST_QUESTION_PATH)
        assert len(test_question_df) == cls.TEST_QUESTION_NUM
        return test_question_df

    @classmethod
    def get_question_category_df(cls):
        question_category_df = pd.read_csv(cls.QUESTION_CATEGORY_PATH)
        assert len(question_category_df) == cls.QUESTION_NUM

        # TODO FIX 临时用补丁修正分类
        classification_bad_case = File.json_load(f"{cls.EXPERIMENT_OUTPUT_DIR}/classification_bad_case.json")
        for category, bad_cases in classification_bad_case.items():
            qids = [case["id"] for case in bad_cases]
            condition = question_category_df["问题id"].isin(qids)
            question_category_df.loc[condition, "分类"] = category
            print(f"[FIX] 将{len(qids)}条问题的分类改为{category}: {qids}")

        return question_category_df

    @classmethod
    def get_tokenizer(cls, model_name: Optional[str] = None, mode: str = ModelMode.EVAL, **kwargs):
        assert mode in ModelMode.values()
        # batch推理需要左padding，chat则不受影响。而训练需要右padding
        padding_side = "left" if mode == ModelMode.EVAL else "right"

        model_dir = cls.model_dir(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, padding_side=padding_side,
                                                  **kwargs)
        tokenizer.pad_token_id = tokenizer.eod_id

        # 添加decode_each、batch_decode_each、cut方法
        tokenizer.decode_each = MethodType(decode_each, tokenizer)
        tokenizer.batch_decode_each = MethodType(batch_decode_each, tokenizer)
        tokenizer.cut = MethodType(cut, tokenizer)

        return tokenizer

    @classmethod
    def get_model(cls, model_name: Optional[str] = None, mode: str = ModelMode.EVAL, device_map: str = "cuda",
                  torch_dtype: Optional[str] = None, use_flash_attn: bool = True, use_dynamic_ntk: bool = True,
                  use_logn_attn: bool = True, generation_config: Optional[Dict[str, str]] = None, **kwargs):
        assert mode in ModelMode.values()

        model_name = model_name or cls.MODEL_NAME
        model_dir = cls.model_dir(model_name)

        if not torch_dtype:
            torch_dtype = torch.float16 if "Int" in model_name else torch.bfloat16
        if torch_dtype == torch.float16:
            fp16, bf16, fp32 = True, False, False
        elif torch_dtype == torch.bfloat16:
            fp16, bf16, fp32 = False, True, False
        elif torch_dtype == torch.float32:
            fp16, bf16, fp32 = False, False, True
        else:
            raise ValueError(f"{torch_dtype=} must be one of torch.float16, torch.bfloat16 or torch.float32")

        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map=device_map, torch_dtype=torch_dtype, fp16=fp16, bf16=bf16,
            fp32=fp32, use_flash_attn=use_flash_attn, use_dynamic_ntk=use_dynamic_ntk, use_logn_attn=use_logn_attn,
            **kwargs)
        assert model.device.type == device_map
        assert model.dtype == torch_dtype

        if use_flash_attn:
            assert hasattr(model.transformer.h[0].attn, "core_attention_flash")
            # rotary_embedding和rms_norm没办法检查

        if "Int" in model_name:
            # 使用exllama插件速度能x7+！！！
            assert "exllama" in type(model.transformer.h[0].mlp.c_proj).__module__

        if mode == ModelMode.EVAL:
            model = model.eval()

        generation_config = generation_config or dict()
        model.generation_config = cls.get_generation_config(model_name, **generation_config)

        # 添加batch方法
        model.batch = MethodType(batch, model)

        return model

    @classmethod
    def get_generation_config(cls, model_name: Optional[str] = None, do_sample: bool = False, **kwargs):
        model_dir = cls.model_dir(model_name)
        if not do_sample:
            # 强制恢复成默认值，防止警告（即使主动输入）
            kwargs.update(top_p=1, top_k=50)

        generation_config = GenerationConfig.from_pretrained(
            model_dir, trust_remote_code=True, do_sample=do_sample, **kwargs)
        return generation_config

    @classmethod
    def set_seed(cls, seed: Optional[int] = None) -> None:
        seed = seed or cls.SEED
        set_seed(seed)

    @classmethod
    def to_dict(cls):
        return {k: v for k, v in vars(cls).items() if
                not k.startswith("_") and not isinstance(v, (classmethod, partialmethod))}
