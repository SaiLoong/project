# -*- coding: utf-8 -*-
# @file generate_sql_test_question.py
# @author zhangshilong
# @date 2024/7/12

from collections import defaultdict
from dataclasses import dataclass
from random import choice
from random import randint
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


@Decorator(
    cluster_label=0,
    question_template="请帮我查询下，在{year}年{month}月的报告中，报告期基金总申购份额和报告期基金总赎回份额差额最大的一只基金的简称是什么？差额有多少？保留两位小数。",
    sql_template="""
    SELECT 基金简称, ROUND(报告期基金总申购份额 - 报告期基金总赎回份额, 2) AS 差额
    FROM 基金规模变动表
    WHERE strftime('%Y%m', 截止日期) = '{year}{month}'
    ORDER BY 差额 DESC
    LIMIT 1;
    """
)
def gen0(year=None, month=None):
    return dict(
        year=year or choice(years),
        month=month or choice(months)
    )


@Decorator(
    cluster_label=1,
    question_template="请帮我查询在截止{year}-{monthday}的基金定期报告中，基金总赎回份额为零的基金有几个？",
    sql_template="""
    SELECT COUNT(DISTINCT(基金代码)) AS 数量
    FROM 基金规模变动表
    WHERE 截止日期 LIKE '{year}-{monthday}%'
    AND 报告类型 = '基金定期报告'
    AND 报告期基金总赎回份额 = 0
    LIMIT 1;
    """
)
def gen1(year=None, monthday=None):
    return dict(
        year=year or choice(years),
        monthday=monthday or choice(monthdays)
    )


@Decorator(
    cluster_label=2,
    question_template="帮我查一下在{year}年，代码为{code}的A股股票今开盘高于昨收盘的天数？",
    sql_template="""
    SELECT COUNT(DISTINCT(交易日)) AS 天数
    FROM A股票日行情表
    WHERE 股票代码 = '{code}'
    AND 交易日 LIKE '{year}%'
    AND [今开盘(元)] > [昨收盘(元)]
    LIMIT 1;  
    """
)
def gen2(year=None, code=None):
    return dict(
        year=year or choice(years),
        code=code or random("A股票日行情表", "股票代码")
    )


@Decorator(
    cluster_label=3,
    question_template="针对{year}年的{report}，有多少家基金的个人投资者持有份额占比不足{percent}%?",
    sql_template="""
    SELECT COUNT(DISTINCT(基金代码)) AS 数量
    FROM 基金份额持有人结构
    WHERE 定期报告所属年度 = {year}
    AND 报告类型 = '{report}'
    AND 个人投资者持有的基金份额占总份额比例 < {percent}
    LIMIT 1;
    """
)
def gen3a(year=None, report=None, percent=None):
    return dict(
        year=year or choice(years),
        report=report or random("基金份额持有人结构", "报告类型"),
        percent=percent or randint(1, 100)
    )


@Decorator(
    cluster_label=3,
    question_template="请帮我查询下，在{year}年{season}季报报告中，{code}基金的第一大重仓可转债同期还有多少只基金也进行了持仓？",
    sql_template="""
    With t1 AS (
        SELECT *
        FROM 基金可转债持仓明细
        WHERE 持仓日期 >= '{year}{start}'
        AND 持仓日期 <= '{year}{end}'
        AND 报告类型 = '季报'
    )
    SELECT COUNT(DISTINCT(基金代码)) AS 数量
    FROM t1
    WHERE 对应股票代码 IN (
        SELECT 对应股票代码
        FROM t1
        WHERE 基金代码 = '{code}'
        AND 第N大重仓股 = 1
    )
    AND 基金代码 != '{code}'
    LIMIT 1;
    """
)
def gen3b(year=None, season=None, code=None):
    season = season or choice(seasons)
    start, end = season_to_start_end[season]

    return dict(
        year=year or choice(years),
        season=season,
        start=start,
        end=end,
        code=code or random("基金可转债持仓明细", "基金代码")
    )


@Decorator(
    cluster_label=4,
    question_template="我想知道{name}基金在{date}的{report}中，其可转债持仓占比最大的是哪个行业？用{standard}一级行业来统计。",
    sql_template="""
    SELECT 一级行业名称
    FROM A股公司行业划分表
    WHERE 股票代码 IN (
        SELECT 对应股票代码
        FROM 基金可转债持仓明细
        WHERE 基金简称 = '{name}'
        AND 持仓日期 = '{date}'
        AND 报告类型 = '{report}'
        ORDER BY 第N大重仓股 ASC
        LIMIT 1
    )
    AND 行业划分标准 LIKE '{standard}%'
    AND 交易日期 = '{date}'
    LIMIT 1;
    """
)
def gen4(name=None, date=None, report=None, standard=None):
    return dict(
        name=name or random("基金可转债持仓明细", "基金简称"),
        date=date or random("基金可转债持仓明细", "持仓日期"),
        report=report or random("基金可转债持仓明细", "报告类型"),
        standard=standard or choice(standards)
    )


# ========================================================================
# 模板

@Decorator(
    cluster_label=NotImplemented,
    question_template="",
    sql_template="""
    SELECT
    FROM
    WHERE
    AND
    AND
    LIMIT 1;
    """
)
def gen(year=None, monthday=None):
    return dict(
        year=year or choice(years),
        monthday=monthday or choice(monthdays)
    )
