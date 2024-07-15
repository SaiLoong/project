# -*- coding: utf-8 -*-
# @file sql_utils.py
# @author zhangshilong
# @date 2024/7/14

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from ..tools.config import Config
from ..tools.utils import String


class GeneratorMeta(type):
    def __init__(cls, name, bases, attr):
        super().__init__(name, bases, attr)
        cls.database = Config.get_database()
        cls.generators = defaultdict(dict)
        cls.aggregation_df = Config.get_sql_question_aggregation_df()


@dataclass
class Generator(metaclass=GeneratorMeta):
    cluster: int
    question_template: str
    sql_template: str
    preprocess_params_function: Callable
    answer_template: str
    postprocess_result_function: Callable

    def __post_init__(self):
        self.sql_template = self.sql_template.replace("\n    ", "\n").strip()
        self.name = self.preprocess_params_function.__name__
        self.generators[self.cluster][self.name] = self
        self.cluster_df = self.get_cluster_df()
        self.questions = self.cluster_df["问题"].tolist()

    def get_cluster_df(self):
        cluster_df = self.aggregation_df.query(f"问题聚类 == {self.cluster}")
        condition = cluster_df["问题"].map(self.parse).astype(bool)
        return cluster_df[condition]

    def __call__(self, **params):
        params = self.preprocess_params_function(**params)
        question = self.question_template.format(**params)
        sql = self.sql_template.format(**params)
        result = self.database.query(sql).to_dict(orient="records")

        self.postprocess_result_function(result)

        return question, sql, result

    def parse(self, question):
        try:
            return String.backstep_format_params(self.question_template, question)
        except ValueError:
            return None

    def query(self, question):
        params = self.parse(question)
        return self(**params)

    def batch_query(self, questions, tqdm_desc=None):
        sqls = list()
        for question in questions:
            params = self.parse(question)
            params = self.preprocess_params_function(**params)
            sqls.append(self.sql_template.format(**params))

        results = [result.to_dict(orient="records") for result in self.database.batch_query(sqls, tqdm_desc=tqdm_desc)]
        return list(zip(questions, sqls, results))

    # TODO ing 输出是否非空、生成submit_json(文件名包括聚类、题目数量、预期分数(小数点后两位))
    def validate(self, verbose=False):
        results = list()
        for question, sql, result in self.batch_query(self.questions, tqdm_desc=f"{self.name}"):
            if verbose:
                print(f"[{self.name}]:\n{question=}\n{sql=}\n{result=}\n")

            if not result:
                print(f"[警告] {self.name}的问题{repr(question)}查询结果为空，SQL:\n{sql}")
            results.append(result)

        print(f"{results=}")


class GeneratorDecorator:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, func):
        return Generator(*self.args, preprocess_params_function=func, **self.kwargs)


# =====================================================================================
# TODO

# 0-15已验证ok
def validate(cls, cluster=None, verbose=False):
    if cluster is None:
        for cluster, generators in cls.generators.items():
            if generators:
                cls.validate(cluster, verbose)
    else:
        questions = cls.questions[cluster]
        generators = cls.generators[cluster].values()

        # 将问题分配给适合的generator
        assign_questions = [list() for _ in range(len(generators))]
        for question in questions:
            for idx, generator in enumerate(generators):
                params = generator.parse(question)
                if params:
                    assign_questions[idx].append(question)
                    break
            else:
                print(f"[警告] 聚类{cluster}的问题{repr(question)}匹配失败")

        # 每个generator批量执行负责的问题，并检查查询结果是否为空
        for generator, _questions in zip(generators, assign_questions):
            for question, sql, result in generator.batch_query(_questions, tqdm_desc=f"聚类{cluster}"):
                if verbose:
                    print(f"[{cluster=}]\n{question=}\n{sql=}\n{result=}\n")

                if not result:
                    print(f"[警告] 聚类{cluster}的问题{repr(question)}查询结果为空，SQL:\n{sql}")
