# -*- coding: utf-8 -*-
# @file generate_sql_test_question.py
# @author zhangshilong
# @date 2024/7/12

from collections import defaultdict
from dataclasses import dataclass
from random import choice
from typing import Callable

from ..tools.config import Config
from ..tools.utils import File
from ..tools.utils import String

# TODO 待整合
#   Generator加方法，一键测试同簇的所有问题（如果不成功，将问题打印出来，不报错）
db = Config.get_database()
db_metadata = File.json_load(Config.DATABASE_METADATA_PATH)
distinct_values_dct = {
    table: {
        record["字段名"]: record["唯一值抽样"]
        for record in v["字段信息"]
    }
    for table, v in db_metadata.items()
}


def choice_from_column(table, column):
    return choice(distinct_values_dct[table][column])


def choice_from_dict(dct):
    return choice(list(dct.keys()))


years = [2019, 2020, 2021]
standards = ["中信", "申万"]
stock_to_table = {
    "A股": "A股票日行情表",
    "港股": "港股票日行情表"
}

# 上面是全局环境
# =====================================================================================================
# 统一规范：
# 1. 为了防止有冗余数据（同时也不想逐条检查），所有COUNT都加上DISTINCT
# 2. 所有sql均以'LIMIT XX;'结尾，哪怕只有一条数据也要加
# 3. 子查询根据需求使用LIMIT, 如果是单条, WHERE用=而不是IN

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
"""
基金基本信息：
    管理费率、托管费率：取值形如 '1.2%', 经测试可以直接比较（如 管理费率 < '0.8%' ）

基金股票持仓明细：
    股票代码：既可能在 A股票日行情表、也可能在 港股票日行情表 上！
    持仓日期：将初步分析，部分 报告类型 = '季报'、'年报(含半年报)' 的基金虽然可能有不在季末的报告，但季末一定有报告。因此取条件时直接以季末为准

基金可转债持仓明细：
    对应股票代码：只有A股，没有港股
    持仓日期：将初步分析，部分 报告类型 = '季报' 的基金虽然可能有不在季末的报告，但季末一定有报告。因此取条件时直接以季末为准

A股票日行情表：
    股票代码：与 港股票日行情表 的 股票代码 没有重叠

港股票日行情表：
    股票代码：与 A股票日行情表 的 股票代码 没有重叠

A股公司行业划分表：
    股票代码：只有A股，没有港股
    行业划分标准=中信/申万行业分类 下，有部分一级行业名称（例如 汽车 计算机）会重叠

基金规模变动表：
    截止日期：月日的组合只有4种可能（季末），连同年一共只有12种取值

基金份额持有人结构：
    个人/机构投资者持有的基金份额占总份额比例：是[0, 100]的数字
"""


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

for x in gen.query(question):
    print(x, "\n\n")

sql = """
SELECT
FROM
WHERE
AND
AND
LIMIT 100;
"""
db.query(sql)

# =====================================================================================================
# 笔记


# TODO 可能有的问题：
#  聚类0、8题意模糊，不确定需不需要考虑绝对值，先不考虑，等钉钉群回复
#  所有用到 基金股票持仓明细、基金可转债持仓明细 报告日期的sql，日期不对（但应该不可能）

# TODO 事后检查： 有没有问题对不上模板、查询结果为空（原因可能为：没考虑港股、报告日期不在月末）
