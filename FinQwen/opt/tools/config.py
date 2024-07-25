# -*- coding: utf-8 -*-
# @file config.py
# @author zhangshilong
# @date 2024/7/6

import platform
from functools import partialmethod
from types import MethodType
from typing import Dict
from typing import Optional
from typing import Union

import pandas as pd
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from transformers import GenerationConfig
from transformers import set_seed

from constant import Category
from constant import ModelMode
from constant import ModelName
from qwen_utils import batch
from qwen_utils import batch_decode_each
from qwen_utils import cut
from qwen_utils import decode_each
from qwen_utils import make_finetune_inputs
from qwen_utils import pairwise_jaccard_distances
from qwen_utils import pairwise_jaccard_scores
from utils import Database
from utils import File


class ConfigMeta(type):
    def __init__(cls, name, bases, attr):
        super().__init__(name, bases, attr)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.max_colwidth", 500)  # 显示文本长些
        plt.rcParams["font.sans-serif"] = ["SimHei"]  # 正确显示中文
        plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号“-”
        tqdm.pandas()  # DataFrame添加progress_apply方法

        cls.set_seed()
        if platform.system() == "Linux":
            [File.makedirs(v) for k, v in vars(cls).items() if k.endswith("_DIR")]


class Config(metaclass=ConfigMeta):
    SEED = 1024
    COMPANY_NUM = 80
    QUESTION_NUM = 1000
    TEXT_QUESTION_NUM = 400
    SQL_QUESTION_NUM = QUESTION_NUM - TEXT_QUESTION_NUM

    CLASSIFICATION_TEST_QUESTION_SAMPLE_NUM = 100
    DATABASE_RECORD_SAMPLE_NUM = 1000
    COLUMN_DISTINCT_VALUE_SAMPLE_NUM = 1000
    SQL_CLUSTER_NUM = 57
    SQL_GENERATOR_NUM = 64
    NL2SQL_TRAIN_DATASET_NUM = SQL_GENERATOR_NUM * 400
    NL2SQL_VALIDATION_DATASET_NUM = SQL_GENERATOR_NUM * 10
    NL2SQL_TEST_DATASET_NUM = SQL_GENERATOR_NUM * 10
    SQL_PROMPT_EXAMPLE_NUM = SQL_GENERATOR_NUM * 10

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

    PREPARE_DIR = File.join(WORKSPACE_DIR, "prepare")
    PREPARE_OUTPUT_DIR = File.join(PREPARE_DIR, "output")
    NL2SQL_FINETUNE_DIR = File.join(PREPARE_OUTPUT_DIR, "nl2sql_finetune")

    INTERMEDIATE_DIR = File.join(WORKSPACE_DIR, "intermediate")
    COMPANY_PATH = File.join(INTERMEDIATE_DIR, "A1_cid_to_company.csv")
    COMPANY_PDF_DIR = File.join(INTERMEDIATE_DIR, "pdf")
    COMPANY_TXT_DIR = File.join(INTERMEDIATE_DIR, "txt")
    CLASSIFICATION_TEST_QUESTION_PATH = File.join(INTERMEDIATE_DIR, "classification_test_question.csv")
    QUESTION_CLASSIFICATION_PATH = File.join(INTERMEDIATE_DIR, "A2_question_classification.csv")
    SQL_QUESTION_AGGREGATION_PATH = File.join(INTERMEDIATE_DIR, "sql_question_aggregation.csv")
    DATABASE_METADATA_PATH = File.join(INTERMEDIATE_DIR, "database_metadata.json")
    SQL_QUESTION_ANSWER_PATH = File.join(INTERMEDIATE_DIR, "sql_question_answer.csv")
    NL2SQL_DATASET_DIR = File.join(INTERMEDIATE_DIR, "nl2sql_dataset")
    NL2SQL_TRAIN_DATASET_PATH = File.join(NL2SQL_DATASET_DIR, "nl2sql_train_dataset.csv")
    NL2SQL_VALIDATION_DATASET_PATH = File.join(NL2SQL_DATASET_DIR, "nl2sql_validation_dataset.csv")
    NL2SQL_TEST_DATASET_PATH = File.join(NL2SQL_DATASET_DIR, "nl2sql_test_dataset.csv")
    SQL_PROMPT_EXAMPLE_PATH = File.join(INTERMEDIATE_DIR, "sql_prompt_example.csv")
    SQL_RESULT_PREDICTION_PATH = File.join(INTERMEDIATE_DIR, "B1_sql_result_prediction.csv")
    SQL_ANSWER_PREDICTION_PATH = File.join(INTERMEDIATE_DIR, "B2_sql_answer_prediction.csv")
    SQL_SUBMIT_RESULT_PATH = File.join(INTERMEDIATE_DIR, "B2_sql_submit_result.jsonl")

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
    def nl2sql_adapter_dir(cls, version: str, step: Union[str, int]):
        adapter_dir = File.join(cls.NL2SQL_FINETUNE_DIR, version, f"checkpoint-{step}")
        assert File.isdir(adapter_dir)
        return adapter_dir

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
    def get_classification_test_question_df(cls):
        classification_test_question_df = pd.read_csv(cls.CLASSIFICATION_TEST_QUESTION_PATH)
        assert len(classification_test_question_df) >= cls.CLASSIFICATION_TEST_QUESTION_SAMPLE_NUM
        return classification_test_question_df

    @classmethod
    def get_question_classification_df(cls):
        question_classification_df = pd.read_csv(cls.QUESTION_CLASSIFICATION_PATH)
        assert len(question_classification_df) == cls.QUESTION_NUM
        assert len(question_classification_df.query(f"问题分类 == '{Category.TEXT}'")) == cls.TEXT_QUESTION_NUM
        return question_classification_df

    @classmethod
    def get_sql_question_aggregation_df(cls):
        sql_question_aggregation_df = pd.read_csv(cls.SQL_QUESTION_AGGREGATION_PATH)
        assert len(sql_question_aggregation_df) == cls.SQL_QUESTION_NUM
        return sql_question_aggregation_df

    @classmethod
    def get_database_metadata(cls):
        return File.json_load(cls.DATABASE_METADATA_PATH)

    @classmethod
    def get_sql_question_answer_df(cls):
        sql_question_answer_df = pd.read_csv(cls.SQL_QUESTION_ANSWER_PATH)
        assert len(sql_question_answer_df) == cls.SQL_QUESTION_NUM
        return sql_question_answer_df

    @classmethod
    def get_nl2sql_dataset(cls):
        dataset = load_dataset("csv", data_files=dict(
            train=cls.NL2SQL_TRAIN_DATASET_PATH,
            validation=cls.NL2SQL_VALIDATION_DATASET_PATH,
            test=cls.NL2SQL_TEST_DATASET_PATH
        ))
        assert dataset.num_rows == dict(
            train=cls.NL2SQL_TRAIN_DATASET_NUM,
            validation=cls.NL2SQL_VALIDATION_DATASET_NUM,
            test=cls.NL2SQL_TEST_DATASET_NUM
        )
        return dataset

    # 发现用一开始生成的10000数据集比25600、6400效果好得多
    @classmethod
    def get_nl2sql_dataset_v2(cls):
        dataset_dir = "/mnt/workspace/intermediate/_nl2sql_dataset_10000"
        dataset = load_dataset("csv", data_files={
            "train": f"{dataset_dir}/nl2sql_train_dataset.csv",  # 10000
            "validation": f"{dataset_dir}/nl2sql_validation_dataset.csv",  # 1000
            "test": f"{dataset_dir}/nl2sql_test_dataset.csv"  # 1000
        })
        return dataset

    @classmethod
    def get_sql_prompt_example_df(cls):
        sql_prompt_example_df = pd.read_csv(cls.SQL_PROMPT_EXAMPLE_PATH)
        assert len(sql_prompt_example_df) == cls.SQL_PROMPT_EXAMPLE_NUM
        return sql_prompt_example_df

    @classmethod
    def get_sql_result_prediction_df(cls):
        sql_result_prediction_df = pd.read_csv(cls.SQL_RESULT_PREDICTION_PATH)
        assert len(sql_result_prediction_df) == cls.SQL_QUESTION_NUM
        return sql_result_prediction_df

    @classmethod
    def get_sql_answer_prediction_df(cls):
        sql_answer_prediction_df = pd.read_csv(cls.SQL_ANSWER_PREDICTION_PATH)
        assert len(sql_answer_prediction_df) == cls.SQL_QUESTION_NUM
        return sql_answer_prediction_df

    @classmethod
    def get_tokenizer(cls, model_name: Optional[str] = None, mode: str = ModelMode.EVAL, **kwargs):
        assert mode in ModelMode.values()
        # batch推理需要左padding，chat则不受影响。而训练需要右padding
        padding_side = "left" if mode == ModelMode.EVAL else "right"

        model_dir = cls.model_dir(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True, padding_side=padding_side,
                                                  **kwargs)
        tokenizer.pad_token_id = tokenizer.eod_id

        # 添加方法
        tokenizer.make_finetune_inputs = MethodType(make_finetune_inputs, tokenizer)
        tokenizer.decode_each = MethodType(decode_each, tokenizer)
        tokenizer.batch_decode_each = MethodType(batch_decode_each, tokenizer)
        tokenizer.cut = MethodType(cut, tokenizer)
        tokenizer.pairwise_jaccard_distances = MethodType(pairwise_jaccard_distances, tokenizer)
        tokenizer.pairwise_jaccard_scores = MethodType(pairwise_jaccard_scores, tokenizer)

        return tokenizer

    @classmethod
    def get_model(cls, model_name: Optional[str] = None, mode: str = ModelMode.EVAL, adapter_dir: Optional[str] = None,
                  device_map: str = "cuda", torch_dtype: Optional[str] = None, use_flash_attn: Optional[bool] = None,
                  use_dynamic_ntk: bool = True, use_logn_attn: bool = True, use_cache: Optional[bool] = None,
                  generation_config: Optional[Dict[str, str]] = None, **kwargs):
        assert mode in ModelMode.values()

        model_name = model_name or cls.MODEL_NAME
        model_dir = cls.model_dir(model_name)

        if not torch_dtype:
            # 量化就算用bf16，推理时还是要先转为fp16
            torch_dtype = torch.float16 if "Int" in model_name else torch.bfloat16
        if torch_dtype == torch.float16:
            fp16, bf16, fp32 = True, False, False
        elif torch_dtype == torch.bfloat16:
            fp16, bf16, fp32 = False, True, False
        elif torch_dtype == torch.float32:
            fp16, bf16, fp32 = False, False, True
        else:
            raise ValueError(f"{torch_dtype=} must be one of torch.float16, torch.bfloat16 or torch.float32")

        if use_flash_attn is None:
            use_flash_attn = "cuda" in device_map

        if use_cache is None:
            use_cache = mode == ModelMode.EVAL

        print(f"loading model {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, trust_remote_code=True, device_map=device_map, torch_dtype=torch_dtype, fp16=fp16, bf16=bf16,
            fp32=fp32, use_flash_attn=use_flash_attn, use_dynamic_ntk=use_dynamic_ntk, use_logn_attn=use_logn_attn,
            use_cache=use_cache, **kwargs)
        assert model.device.type == device_map
        assert model.dtype == torch_dtype

        if use_flash_attn:
            assert hasattr(model.transformer.h[0].attn, "core_attention_flash")
            # rotary_embedding和rms_norm没办法检查

        if "Int4" in model_name:
            # 使用exllama插件速度能x7+！！！但Int8用不了
            assert "exllama" in type(model.transformer.h[0].mlp.c_proj).__module__

        model.train(mode == ModelMode.TRAIN)

        # generation_config只在generate时使用，对训练没有影响
        generation_config = generation_config or dict()
        model.generation_config = cls.get_generation_config(model_name, **generation_config)

        # 添加batch方法
        model.batch = MethodType(batch, model)

        # 用户需要保证model和adapter是配对的
        if adapter_dir:
            model = PeftModel.from_pretrained(model, model_id=adapter_dir)
            model.merge_and_unload()
            print(f"merge_and_unload {adapter_dir} successfully")

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
    def get_database(cls):
        return Database(cls.DATABASE_PATH)

    @classmethod
    def export_submit_result(cls, df: pd.DataFrame, path: str):
        assert path.endswith("submit_result.jsonl")

        question_df = cls.get_question_df()
        submit_df = pd.merge(question_df, df[["问题id", "答案"]], how="left", on="问题id")
        submit_df.fillna("", inplace=True)
        submit_df.rename(columns={"问题id": "id", "问题": "question", "答案": "answer"}, inplace=True)
        File.dataframe_to_jsonl(submit_df, path)

    @classmethod
    def set_seed(cls, seed: Optional[int] = None) -> None:
        seed = seed or cls.SEED
        set_seed(seed)

    @classmethod
    def to_dict(cls):
        return {k: v for k, v in vars(cls).items() if
                not k.startswith("_") and not isinstance(v, (classmethod, partialmethod))}
