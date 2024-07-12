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
    question_template="帮我查一下在{year}年，代码为{code}的{stock}股票今开盘{compare}于昨收盘的天数？",
    sql_template="""
    SELECT COUNT(DISTINCT(交易日)) AS 天数
    FROM {table}
    WHERE 股票代码 = '{code}'
    AND 交易日 LIKE '{year}%'
    AND [今开盘(元)] {sign} [昨收盘(元)]
    LIMIT 1;  
    """
)
def gen2(year=None, code=None, stock=None, compare=None):
    stock = stock or choice(stocks)
    table = stock_to_table[stock]
    compare = compare or choice(compares)
    return dict(
        year=year or choice(years),
        code=code or random(table, "股票代码"),
        stock=stock,
        compare=compare,
        table=table,
        sign=compare_to_sign[compare]
    )


@Decorator(
    cluster_label=3,
    question_template="针对{year}年的{report}，有多少家基金的{unit}投资者持有份额占比不足{percent}%?",
    sql_template="""
    SELECT COUNT(DISTINCT(基金代码)) AS 数量
    FROM 基金份额持有人结构
    WHERE 定期报告所属年度 = {year}
    AND 报告类型 = '{report}'
    AND {unit}投资者持有的基金份额占总份额比例 < {percent}
    LIMIT 1;
    """
)
def gen3a(year=None, report=None, unit=None, percent=None):
    return dict(
        year=year or choice(years),
        report=report or random("基金份额持有人结构", "报告类型"),
        unit=unit or choice(units),
        percent=percent or randint(1, 100)
    )


@Decorator(
    cluster_label=3,
    question_template="请帮我查询下，在{year}年{season}季报报告中，{code}基金的第{numzh}大重仓可转债同期还有多少只基金也进行了持仓？",
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
        AND 第N大重仓股 = {num}
    )
    AND 基金代码 != '{code}'
    LIMIT 1;
    """
)
def gen3b(year=None, season=None, code=None, numzh=None):
    season = season or choice(seasons)
    start, end = season_to_start_end[season]
    numzh = numzh or choice(numzhs)
    num = numzh_to_num[numzh]
    return dict(
        year=year or choice(years),
        season=season,
        code=code or random("基金可转债持仓明细", "基金代码"),
        numzh=numzh,
        start=start,
        end=end,
        num=num
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
    table = "基金可转债持仓明细"
    return dict(
        name=name or random(table, "基金简称"),
        date=date or random(table, "持仓日期"),
        report=report or random(table, "报告类型"),
        standard=standard or choice(standards)
    )


@Decorator(
    cluster_label=5,
    question_template="在{date}的{report}中，{name}基金的债券持仓,其持有最大仓位的债券类型是什么?",
    sql_template="""
    SELECT 债券类型
    FROM 基金债券持仓明细
    WHERE 持仓日期 = '{date}'
    AND 报告类型 = '{report}'
    AND 基金简称 = '{name}'
    ORDER BY 第N大重仓股 ASC
    LIMIT 1;
    """
)
def gen5(date=None, report=None, name=None):
    table = "基金债券持仓明细"
    return dict(
        date=date or random(table, "持仓日期"),
        report=report or random(table, "报告类型"),
        name=name or random(table, "基金简称")
    )


@Decorator(
    cluster_label=6,
    question_template="{manager}管理的{category}产品的数量有多少?",
    sql_template="""
    SELECT COUNT(DISTINCT(基金代码)) AS 数量
    FROM 基金基本信息
    WHERE 管理人 = '{manager}'
    AND 基金类型 = '{category}'
    LIMIT 1;
    """
)
def gen6(manager=None, category=None):
    table = "基金基本信息"
    return dict(
        manager=manager or random(table, "管理人"),
        category=category or random(table, "基金类型")
    )


@Decorator(
    cluster_label=7,
    question_template="{date}日，一级行业为{industry1}的股票的成交量合计是多少？取整。",
    sql_template="""
    SELECT CAST(SUM([成交量(股)]) AS INTEGER) AS 成交量合计
    FROM A股票日行情表
    WHERE 交易日 = '{date}'
    AND 股票代码 IN (
        SELECT 股票代码
        FROM A股公司行业划分表
        WHERE 交易日期 = '{date}'
        AND 一级行业名称 = '{industry1}'
    )
    LIMIT 1;
    """
)
def gen7a(date=None, industry1=None):
    table = "A股公司行业划分表"
    return dict(
        date=date or random(table, "交易日期"),
        industry1=industry1 or random(table, "一级行业名称")
    )


@Decorator(
    cluster_label=7,
    question_template="请帮我计算：在{date}，日收益率为{compare}的{stock}股票有几个。",
    sql_template="""
    SELECT COUNT(DISTINCT(股票代码)) AS 数量
    FROM {table}
    WHERE 交易日 = '{date}'
    AND [收盘价(元)] {sign} [昨收盘(元)]
    LIMIT 1;
    """
)
def gen7b(date=None, compare=None, stock=None):
    compare = compare or choice(compares)
    stock = stock or choice(stocks)
    table = stock_to_table[stock]
    return dict(
        date=date or random(table, "交易日"),
        compare=compare,
        stock=stock,
        table=table,
        sign=compare_to_sign[compare]
    )
