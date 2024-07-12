# -*- coding: utf-8 -*-
# @file sql_generator.py
# @author zhangshilong
# @date 2024/7/13

# TODO 临时存放处理好的Generator

from random import randint

from generate_sql_test_question import *


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
