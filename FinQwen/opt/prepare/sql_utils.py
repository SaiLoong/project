# -*- coding: utf-8 -*-
# @file sql_utils.py
# @author zhangshilong
# @date 2024/7/14

from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from ..tools.config import Config
from ..tools.utils import String


@dataclass
class Generator:
    question_template: str
    sql_template: str
    preproccess_params_function: Callable

    def __post_init__(self):
        self.sql_template = self.sql_template.replace("\n    ", "\n").strip()
        self.database = Config.get_database()

    def __call__(self, **params):
        params = self.preproccess_params_function(**params)

        question = self.question_template.format(**params)
        sql = self.sql_template.format(**params)
        result = self.database.query(sql).to_dict(orient="records")
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
            params = self.preproccess_params_function(**params)
            sqls.append(self.sql_template.format(**params))

        results = [result.to_dict(orient="records") for result in self.database.batch_query(sqls, tqdm_desc=tqdm_desc)]
        return list(zip(questions, sqls, results))


class ManagerMeta(type):
    def __init__(cls, name, bases, attr):
        super().__init__(name, bases, attr)
        cls.generators = defaultdict(list)
        cls.aggregation_df = Config.get_sql_question_aggregation_df()
        cls.questions = {cluster: df["问题"].tolist() for cluster, df in cls.aggregation_df.groupby("问题聚类")}


@dataclass
class Manager(metaclass=ManagerMeta):
    cluster: int
    question_template: str
    sql_template: str

    def __call__(self, func: Callable):
        generator = Generator(self.question_template, self.sql_template, func)
        self.generators[self.cluster].append(generator)
        return generator

    # 0-13已验证ok
    @classmethod
    def validate(cls, cluster=None, verbose=False):
        if cluster is None:
            for cluster in cls.generators.keys():
                cls.validate(cluster, verbose)
        else:
            questions = cls.questions[cluster]
            generators = cls.generators[cluster]

            # 将问题分配给适合的generator
            assign_questions = [list() for _ in range(len(generators))]
            for question in questions:
                for idx, generator in enumerate(generators):
                    params = generator.parse(question)
                    if params:
                        assign_questions[idx].append(question)
                        break
                else:
                    print(f"聚类{cluster}的问题{repr(question)}匹配失败")

            # 每个generator批量执行负责的问题，并检查查询结果是否为空
            for generator, _questions in zip(generators, assign_questions):
                for question, sql, result in generator.batch_query(_questions, tqdm_desc=f"聚类{cluster}"):
                    if verbose:
                        print(f"[{cluster=}]\n{question=}\n{sql=}\n{result=}\n")

                    if not result:
                        print(f"聚类{cluster}的问题{repr(question)}查询结果为空，SQL:\n{sql}")

# TODO
#  1. Generator从gen3b函数中提取name，然后Manager在注册generators时发现name重叠就删掉旧generator
