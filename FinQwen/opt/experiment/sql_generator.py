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
    WHERE 截止日期 LIKE '{year}-{month}%'
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
    question_template="帮我查一下在{year}年，代码为{code}的{stock}股票今开盘{compare}昨收盘的天数？",
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
    compare = compare or choice(compares_v1)
    return dict(
        year=year or choice(years),
        code=code or random(table, "股票代码"),
        stock=stock,
        compare=compare,
        table=table,
        sign=compare_to_sign_v1[compare]
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
    table = "基金份额持有人结构"
    return dict(
        year=year or choice(years),
        report=report or random(table, "报告类型"),
        unit=unit or choice(units),
        percent=percent or randint(1, 100)
    )


@Decorator(
    cluster_label=3,
    question_template="请帮我查询下，在{year}年{season}季报报告中，{code}基金的第{numzh}大重仓可转债同期还有多少只基金也进行了持仓？",
    # 问题中包含”还有“，因此要扣除自己
    sql_template="""
    WITH t1 AS (
        SELECT *
        FROM 基金可转债持仓明细
        WHERE 持仓日期 BETWEEN '{year}{start}' AND '{year}{end}'
        AND 报告类型 = '季报'
    )
    SELECT COUNT(DISTINCT(基金代码)) AS 数量
    FROM t1
    WHERE 对应股票代码 = (
        SELECT 对应股票代码
        FROM t1
        WHERE 基金代码 = '{code}'
        AND 第N大重仓股 = {num}
        LIMIT 1
    )
    AND 基金代码 != '{code}'
    LIMIT 1;
    """
)
def gen3b(year=None, season=None, code=None, numzh=None):
    season = season or choice(seasons_v1)
    start, end = season_to_start_end_v1[season]
    numzh = numzh or choice(numzhs)
    table = "基金可转债持仓明细"
    return dict(
        year=year or choice(years),
        season=season,
        code=code or random(table, "基金代码"),
        numzh=numzh,
        start=start,
        end=end,
        num=numzh_to_num[numzh]
    )


@Decorator(
    cluster_label=4,
    question_template="我想知道{name}基金在{date}的{report}中，其可转债持仓占比最大的是哪个行业？用{standard}一级行业来统计。",
    sql_template="""
    SELECT 一级行业名称
    FROM A股公司行业划分表
    WHERE 股票代码 = (
        SELECT 对应股票代码
        FROM 基金可转债持仓明细
        WHERE 基金简称 = '{name}'
        AND 持仓日期 = '{date}'
        AND 报告类型 = '{report}'
        ORDER BY 第N大重仓股 ASC
        LIMIT 1
    )
    AND 行业划分标准 = '{standard}行业分类'
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
    # 问题没有明确是中信还是申万标准，有的只在一边，有的两边都有，不过不影响结果
    question_template="{date}日，一级行业为{industry1}的股票的{target}合计是多少？取整。",
    sql_template="""
    SELECT CAST(SUM([{column}]) AS INTEGER) AS {target}合计
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
def gen7a(date=None, industry1=None, target=None):
    target = target or choice(targets)
    table = "A股公司行业划分表"
    return dict(
        date=date or random(table, "交易日期"),
        industry1=industry1 or random(table, "一级行业名称"),
        target=target,
        column=target_to_column[target]
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
    compare = compare or choice(compares_v2)
    stock = stock or choice(stocks)
    table = stock_to_table[stock]
    return dict(
        date=date or random(table, "交易日"),
        compare=compare,
        stock=stock,
        table=table,
        sign=compare_to_sign_v2[compare]
    )


@Decorator(
    cluster_label=8,
    question_template="请帮我计算，在{date}，{standard}行业分类划分的一级行业为{industry1}行业中，涨跌幅最大股票的股票代码是？涨跌幅是多少？百分数保留两位小数。股票涨跌幅定义为：（收盘价 - 前一日收盘价 / 前一日收盘价）* 100%。",
    sql_template="""
    SELECT 股票代码, ROUND(涨跌幅1 * 100, 2) || '%' AS 涨跌幅
    FROM (
        SELECT 股票代码, ([收盘价(元)] / [昨收盘(元)] - 1) AS 涨跌幅1
        FROM A股票日行情表
        WHERE 股票代码 IN (
            SELECT 股票代码
            FROM A股公司行业划分表
            WHERE 交易日期 = '{date}'
            AND 行业划分标准 = '{standard}行业分类'
            AND 一级行业名称 = '{industry1}'
        )
        AND 交易日 = '{date}'
        ORDER BY 涨跌幅1 DESC
    )
    LIMIT 1;
    """
)
def gen8(date=None, standard=None, industry1=None):
    table = "A股公司行业划分表"
    return dict(
        date=date or random(table, "交易日期"),
        standard=standard or choice(standards),
        industry1=industry1 or random(table, "一级行业名称")
    )


@Decorator(
    cluster_label=9,
    question_template="我想知道{company}在{year}年成立了多少只管理费率{compare}于{percent}%的基金？",
    sql_template="""
    SELECT COUNT(DISTINCT(基金代码)) AS 数量
    FROM 基金基本信息
    WHERE 管理人 = '{company}'
    AND 成立日期 LIKE '{year}%'
    AND 管理费率 {sign} '{percent}%'
    LIMIT 1;
    """
)
def gen9(company=None, year=None, compare=None, percent=None):
    compare = compare or choice(compares_v3)
    table = "基金基本信息"
    return dict(
        company=company or random(table, "管理人"),
        year=year or choice(years),
        compare=compare,
        percent=percent or randint(1, 9) / 10,
        sign=compare_to_sign_v3[compare]
    )


@Decorator(
    cluster_label=10,
    question_template="{date}港股{compare}的股票家数有多少家?",
    sql_template="""
    SELECT COUNT(DISTINCT(股票代码)) AS 数量
    FROM 港股票日行情表
    WHERE 交易日 = '{date}'
    AND [收盘价(元)] {sign} [今开盘(元)]
    LIMIT 1;
    """
)
def gen10(date=None, compare=None):
    compare = compare or choice(compares_v4)
    table = "港股票日行情表"
    return dict(
        date=date or random(table, "交易日"),
        compare=compare,
        sign=compare_to_sign_v4[compare]
    )


@Decorator(
    cluster_label=11,
    question_template="我想了解{name}基金,在{year}年{season}的季报第{rank}大重股。该持仓股票当个季度的涨跌幅?请四舍五入保留百分比到小数点两位。",
    # 先把股票找出来，存到t1表中（只有一行数据），股票可能在A股表也可能在港股表，SQL貌似不支持动态选择表的操作，因此只能分别查询A股和港股表，
    # 然后将答案union起来得到t4表（有且只有1个表有数据）。t4表包含目标股票在目标季度内的所有数据，分别找到最早的昨收价和最晚的收盘价（不一定是季度边缘），
    # 存到t5、t6表，最后就能算季度的涨跌幅了（join左右子表都只有1条数据，因此不用写on）
    sql_template="""
    WITH t1 AS (
        SELECT 股票代码
        FROM 基金股票持仓明细
        WHERE 基金简称 = '{name}'
        AND 持仓日期 BETWEEN '{year}{start}' AND '{year}{end}'
        AND 报告类型 = '季报'
        AND 第N大重仓股 = {rank}
        LIMIT 1
    ),
    t4 AS (
        SELECT *
        FROM A股票日行情表 t2 JOIN t1
        ON t2.股票代码 = t1.股票代码
        AND 交易日 BETWEEN '{year}{start}' AND '{year}{end}'
        UNION
        SELECT *
        FROM 港股票日行情表 t3 JOIN t1
        ON t3.股票代码 = t1.股票代码
        AND 交易日 BETWEEN '{year}{start}' AND '{year}{end}'
    ),
    t5 AS (
        SELECT [昨收盘(元)]
        FROM t4
        ORDER BY 交易日 ASC
        LIMIT 1
    ),
    t6 AS (
        SELECT [收盘价(元)]
        FROM t4
        ORDER BY 交易日 DESC
        LIMIT 1
    )
    SELECT ROUND((t6.[收盘价(元)] / t5.[昨收盘(元)] - 1) * 100, 2) || '%' AS 涨跌幅
    FROM t5 JOIN t6
    LIMIT 1;
    """
)
def gen11(name=None, year=None, season=None, rank=None):
    season = season or choice(seasons_v2)
    start, end = season_to_start_end_v2[season]
    table = "基金股票持仓明细"
    return dict(
        name=name or random(table, "基金简称"),
        year=year or choice(years),
        season=season,
        rank=rank or randint(1, 9),
        start=start,
        end=end
    )
