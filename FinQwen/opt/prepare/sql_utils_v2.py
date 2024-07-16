# -*- coding: utf-8 -*-
# @file sql_utils_v2.py
# @author zhangshilong
# @date 2024/7/15

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Union

import pandas as pd

from ..tools.config import Config
from ..tools.utils import File
from ..tools.utils import String


class Record(NamedTuple):
    params: Dict[str, Union[str, int]]
    question: str
    sql: str
    result: List[Dict[str, Union[str, int]]]
    answer: str


class Manager:
    generators = defaultdict(dict)


@dataclass
class Generator:
    cluster: int
    question_template: str
    sql_template: str
    answer_template: str
    verification_score: float = None

    def __post_init__(self):
        self.name = self.__class__.__name__
        m = re.fullmatch(f"Generator({self.cluster}[a-z]?)", self.name)
        assert m, f"类名{self.name}不符合规范"
        self.abbr = m.group(1)

        assert " \n" not in self.sql_template, "SQL模板行末存在空格"
        self.sql_template = self.sql_template.replace("\n    ", "\n").strip()

        self.database = Config.get_database()
        self.question_df = Config.get_question_df()
        self.aggregation_df = Config.get_sql_question_aggregation_df()
        self.cluster_df = self._get_cluster_df()
        self.expectation_score = round(100 / 600 * len(self.cluster_df), 2)
        self.questions = self.cluster_df["问题"].tolist()
        self.records = None

        Manager.generators[self.cluster][self.name] = self

    def _get_cluster_df(self):
        cluster_df = self.aggregation_df.query(f"问题聚类 == {self.cluster}")
        condition = cluster_df["问题"].map(self.parse).astype(bool)
        return cluster_df[condition].reset_index(drop=True)

    def preprocess_params(self, **params):
        raise NotImplementedError

    def postprocess_result(self, result, params):
        if result:
            return result[0]
        return None

    def __call__(self, **params):
        params = self.preprocess_params(**params)
        question = self.question_template.format(**params)
        sql = self.sql_template.format(**params)
        result = self.database.query(sql).to_dict(orient="records")

        result2 = self.postprocess_result(result, params)
        answer = self.answer_template.format(**params, **result2) if result2 else None
        return Record(params, question, sql, result, answer)

    def parse(self, question):
        try:
            return String.backstep_format_params(self.question_template, question)
        except ValueError:
            return None

    def query(self, question):
        params = self.parse(question)
        return self(**params)

    def batch_query(self, questions):
        params_list = list()
        sqls = list()
        for question in questions:
            params = self.parse(question)
            params = self.preprocess_params(**params)
            params_list.append(params)
            sqls.append(self.sql_template.format(**params))

        records = list()
        raw_results = self.database.batch_query(sqls)
        for params, question, sql, raw_result in zip(params_list, questions, sqls, raw_results):
            result = raw_result.to_dict(orient="records")
            result2 = self.postprocess_result(result, params)
            answer = self.answer_template.format(**params, **result2) if result2 else None
            records.append(Record(params, question, sql, result, answer))
        return records

    def refresh_records(self):
        if not self.records:
            self.records = self.batch_query(self.questions)
            for record in self.records:
                if not record.answer:
                    print(f"[警告！] 问题（{repr(record.question)}）没有查询到答案！SQL:\n{record.sql}")

            self.cluster_df["答案"] = [record.answer for record in self.records]
        return self.records

    @property
    def results(self):
        self.refresh_records()
        return [record.result for record in self.records]

    def export(self):
        self.refresh_records()
        df = pd.merge(self.question_df, self.cluster_df[["问题id", "答案"]], how="left", on="问题id")
        df.fillna("", inplace=True)
        df.rename(columns={"问题id": "id", "问题": "question", "答案": "answer"}, inplace=True)

        score = str(self.expectation_score).replace(".", "p")
        generator_dir = f"{Config.PREPARE_OUTPUT_DIR}/generator"
        File.makedirs(generator_dir)
        File.dataframe_to_jsonl(df, f"{generator_dir}/{self.abbr}_{len(self.cluster_df)}_{score}_submit_result.jsonl")

# TODO
#  1. 把生成answer的功能加上，因此需要重构utils
#  2. 编写自动生成提交文件的方法，要考虑和组合
#  3. 各种读写文件集成到File ?
#  c0拿到1.15分（满分1.17），说明模板是正确的
