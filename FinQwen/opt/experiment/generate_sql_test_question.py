# -*- coding: utf-8 -*-
# @file generate_sql_test_question.py
# @author zhangshilong
# @date 2024/7/12

from collections import defaultdict
from dataclasses import dataclass
from random import choice
from typing import Callable

import pandas as pd

from ..tools.config import Config
from ..tools.utils import File
from ..tools.utils import String

db = Config.get_database()

# TODO 先不集成到Config/Database中，再观察一下
db_metadata = File.json_load(Config.DATABASE_METADATA_PATH)
column_dfs = {table: pd.DataFrame(v["字段信息"]) for table, v in db_metadata.items()}


def get_distinct_values(table, column):
    return column_dfs[table].query(f"字段名 == '{column}'")["唯一值抽样"].iloc[0]


def random(table, column):
    return choice(get_distinct_values(table, column))


years = [2019, 2020, 2021]
month_to_monthday = {
    "03": "03-31",
    "06": "06-30",
    "09": "09-30",
    "12": "12-31"
}
months = list(month_to_monthday.keys())
monthdays = list(month_to_monthday.values())
season_to_start_end = {
    "Q1": ("0101", "0331"),
    "Q2": ("0401", "0630"),
    "Q3": ("0701", "0930"),
    "Q4": ("1001", "1231")
}
seasons = list(season_to_start_end.keys())
standards = ["中信", "申万"]
compare_to_sign_v1 = {
    "高": ">",
    "低": "<"
}
compares_v1 = list(compare_to_sign_v1.keys())
compare_to_sign_v2 = {
    "正": ">",
    "负": "<"
}
compares_v2 = list(compare_to_sign_v2.keys())
compare_to_sign_v3 = {
    "大": ">",
    "小": "<"
}
compares_v3 = list(compare_to_sign_v3.keys())
compare_to_sign_v4 = {
    "上升": ">",
    "下跌": "<"
}
compares_v4 = list(compare_to_sign_v4.keys())
stock_to_table = {
    "A股": "A股票日行情表",
    "港股": "港股票日行情表"
}
stocks = list(stock_to_table.keys())
units = ["个人", "机构"]
numzh_to_num = {
    "一": 1,
    "二": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9
}
numzhs = list(numzh_to_num.keys())

# 上面是全局环境
# =====================================================================================================
# 统一规范：
# 1. 为了防止有冗余数据（同时也不想逐条检查），所有COUNT都加上DISTINCT
# 2. 所有sql均以'LIMIT XX;'结尾，哪怕只有一条数据也要加

# TODO 变成Generator或Decorator的类属性？
generators = defaultdict(list)


@dataclass
class Generator:
    question_template: str
    sql_template: str
    preproccess_params: Callable

    def __post_init__(self):
        self.sql_template = self.sql_template.replace("\n    ", "\n").strip()

    def __call__(self, **params):
        params = self.preproccess_params(**params)

        question = self.question_template.format(**params)
        sql = self.sql_template.format(**params)
        result = db.query(sql).to_dict(orient="records")
        return question, sql, result

    def parse(self, question):
        params = String.backstep_format_params(self.question_template, question)
        assert self.question_template.format(**params) == question
        return params

    def query(self, question):
        params = self.parse(question)
        return self(**params)


@dataclass
class Decorator:
    cluster_label: int
    question_template: str
    sql_template: str

    def __call__(self, func: Callable):
        generator = Generator(self.question_template, self.sql_template, func)
        generators[self.cluster_label].append(generator)
        return generator


# =====================================================================================================
# 之前的generator先放到sql_generator.py


# =====================================================================================================
# 模板


@Decorator(
    cluster_label=NotImplemented,
    question_template="",
    sql_template=""
)
def gen():
    table = ""
    return dict(

    )


gen()

question = ""
gen.query(question)

sql1 = """
SELECT
FROM
WHERE
AND
AND
LIMIT 100;
"""
db.query(sql1)

# TODO 聚类0、8题意模糊，不确定需不需要考虑绝对值，先不考虑，等钉钉群回复
