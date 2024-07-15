# -*- coding: utf-8 -*-
# @file sql_generator.py
# @author zhangshilong
# @date 2024/7/13

# TODO 临时存放处理好的Generator

from random import choice
from random import randint

from sql_utils import Manager
from ..tools.config import Config

db_metadata = Config.get_database_metadata()
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
rankzh_to_rank = {
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
target_to_column = {
    "成交量": "成交量(股)",
    "成交金额": "成交金额(元)"
}


@Manager(
    cluster=0,
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
    months = ["03", "06", "09", "12"]

    return dict(
        year=year or choice(years),
        month=month or choice(months)
    )


@Manager(
    cluster=1,
    question_template="请帮我查询在截止{year}-{monthday}的基金定期报告中，基金总赎回份额为零的基金有几个？",
    sql_template="""
    SELECT COUNT(基金代码) AS 数量
    FROM 基金规模变动表
    WHERE 截止日期 = '{year}-{monthday} 00:00:00'
    AND 报告类型 = '基金定期报告'
    AND 报告期基金总赎回份额 = 0
    LIMIT 1;
    """
)
def gen1(year=None, monthday=None):
    monthdays = ["03-31", "06-30", "09-30", "12-31"]

    return dict(
        year=year or choice(years),
        monthday=monthday or choice(monthdays)
    )


@Manager(
    cluster=2,
    question_template="帮我查一下在{year}年，代码为{code}的{stock}股票今开盘{compare}昨收盘的天数？",
    sql_template="""
    SELECT COUNT(交易日) AS 天数
    FROM {table}
    WHERE 股票代码 = '{code}'
    AND 交易日 LIKE '{year}%'
    AND [今开盘(元)] {sign} [昨收盘(元)]
    LIMIT 1;  
    """
)
def gen2(year=None, code=None, stock=None, compare=None):
    stock = stock or choice_from_dict(stock_to_table)
    table = stock_to_table[stock]

    compare_to_sign = {
        "高于": ">",
        "低于": "<"
    }
    compare = compare or choice_from_dict(compare_to_sign)

    return dict(
        year=year or choice(years),
        code=code or choice_from_column(table, "股票代码"),
        stock=stock,
        compare=compare,
        table=table,
        sign=compare_to_sign[compare]
    )


@Manager(
    cluster=3,
    question_template="针对{year}年的{report}，有多少家基金的{role}持有份额占比不足{percent}%?",
    sql_template="""
    SELECT COUNT(基金代码) AS 数量
    FROM 基金份额持有人结构
    WHERE 定期报告所属年度 = {year}
    AND 报告类型 = '{report}'
    AND {role}持有的基金份额占总份额比例 < {percent}
    LIMIT 1;
    """
)
def gen3a(year=None, report=None, role=None, percent=None):
    roles = ["个人投资者", "机构投资者"]
    table = "基金份额持有人结构"

    return dict(
        year=year or choice(years),
        report=report or choice_from_column(table, "报告类型"),
        role=role or choice(roles),
        percent=percent or randint(1, 100)
    )


@Manager(
    cluster=3,
    question_template="请帮我查询下，在{year}年{season}季报报告中，{code}基金的第{rankzh}大重仓可转债同期还有多少只基金也进行了持仓？",
    # “WHERE 对应股票代码 = (..)”里共限定了四个条件，其实已经保证必然只有一条记录。外面有三个条件限制，因此 基金代码 必然唯一
    # 问题是问“还有”，因此要扣除自己
    sql_template="""
    WITH t1 AS (
        SELECT *
        FROM 基金可转债持仓明细
        WHERE 持仓日期 = '{year}{monthday}'
        AND 报告类型 = '季报'
    )
    SELECT COUNT(基金代码) AS 数量
    FROM t1
    WHERE 对应股票代码 = (
        SELECT 对应股票代码
        FROM t1
        WHERE 基金代码 = '{code}'
        AND 第N大重仓股 = {rank}
        LIMIT 1
    )
    AND 基金代码 != '{code}'
    LIMIT 1;
    """
)
def gen3b(year=None, season=None, code=None, rankzh=None):
    season_to_monthday = {
        "Q1": "0331",
        "Q2": "0630",
        "Q3": "0930",
        "Q4": "1231"
    }
    season = season or choice_from_dict(season_to_monthday)
    monthday = season_to_monthday[season]

    rankzh = rankzh or choice_from_dict(rankzh_to_rank)
    table = "基金可转债持仓明细"

    return dict(
        year=year or choice(years),
        season=season,
        code=code or choice_from_column(table, "基金代码"),
        rankzh=rankzh,
        monthday=monthday,
        rank=rankzh_to_rank[rankzh]
    )


@Manager(
    cluster=4,
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
        name=name or choice_from_column(table, "基金简称"),
        date=date or choice_from_column(table, "持仓日期"),
        report=report or choice_from_column(table, "报告类型"),
        standard=standard or choice(standards)
    )


@Manager(
    cluster=5,
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
        date=date or choice_from_column(table, "持仓日期"),
        report=report or choice_from_column(table, "报告类型"),
        name=name or choice_from_column(table, "基金简称")
    )


@Manager(
    cluster=6,
    question_template="{manager}管理的{category}产品的数量有多少?",
    sql_template="""
    SELECT COUNT(基金代码) AS 数量
    FROM 基金基本信息
    WHERE 管理人 = '{manager}'
    AND 基金类型 = '{category}'
    LIMIT 1;
    """
)
def gen6(manager=None, category=None):
    table = "基金基本信息"

    return dict(
        manager=manager or choice_from_column(table, "管理人"),
        category=category or choice_from_column(table, "基金类型")
    )


@Manager(
    cluster=7,
    question_template="{date}日，一级行业为{industry1}的股票的{target}合计是多少？取整。",
    # 问题没有明确是中信还是申万标准，有的一级行业只有一边有，有的两边都有，不过不影响结果
    # A股票日行情表 里，股票代码 + 交易日 是唯一的
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
    target = target or choice_from_dict(target_to_column)
    table = "A股公司行业划分表"

    return dict(
        date=date or choice_from_column(table, "交易日期"),
        industry1=industry1 or choice_from_column(table, "一级行业名称"),
        target=target,
        column=target_to_column[target]
    )


@Manager(
    cluster=7,
    question_template="请帮我计算：在{date}，日收益率为{compare}的{stock}股票有几个。",
    sql_template="""
    SELECT COUNT(股票代码) AS 数量
    FROM {table}
    WHERE 交易日 = '{date}'
    AND [收盘价(元)] {sign} [昨收盘(元)]
    LIMIT 1;
    """
)
def gen7b(date=None, compare=None, stock=None):
    compare_to_sign = {
        "正": ">",
        "负": "<"
    }
    compare = compare or choice_from_dict(compare_to_sign)

    stock = stock or choice_from_dict(stock_to_table)
    table = stock_to_table[stock]

    return dict(
        date=date or choice_from_column(table, "交易日"),
        compare=compare,
        stock=stock,
        table=table,
        sign=compare_to_sign[compare]
    )


@Manager(
    cluster=8,
    question_template="请帮我计算，在{date}，{standard}行业分类划分的一级行业为{industry1}行业中，涨跌幅最大股票的股票代码是？涨跌幅是多少？百分数保留两位小数。股票涨跌幅定义为：（收盘价 - 前一日收盘价 / 前一日收盘价）* 100%。",
    sql_template="""
    SELECT 股票代码, ROUND(涨跌幅0 * 100, 2) || '%' AS 涨跌幅
    FROM (
        SELECT 股票代码, ([收盘价(元)] / [昨收盘(元)] - 1) AS 涨跌幅0
        FROM A股票日行情表
        WHERE 股票代码 IN (
            SELECT 股票代码
            FROM A股公司行业划分表
            WHERE 交易日期 = '{date}'
            AND 行业划分标准 = '{standard}行业分类'
            AND 一级行业名称 = '{industry1}'
        )
        AND 交易日 = '{date}'
        ORDER BY 涨跌幅0 DESC
    )
    LIMIT 1;
    """
)
def gen8(date=None, standard=None, industry1=None):
    table = "A股公司行业划分表"

    return dict(
        date=date or choice_from_column(table, "交易日期"),
        standard=standard or choice(standards),
        industry1=industry1 or choice_from_column(table, "一级行业名称")
    )


@Manager(
    cluster=9,
    question_template="我想知道{company}在{year}年成立了多少只管理费率{compare}于{percent}%的基金？",
    sql_template="""
    SELECT COUNT(基金代码) AS 数量
    FROM 基金基本信息
    WHERE 管理人 = '{company}'
    AND 成立日期 LIKE '{year}%'
    AND 管理费率 {sign} '{percent}%'
    LIMIT 1;
    """
)
def gen9(company=None, year=None, compare=None, percent=None):
    compare_to_sign = {
        "大": ">",
        "小": "<"
    }
    compare = compare or choice_from_dict(compare_to_sign)
    table = "基金基本信息"

    return dict(
        company=company or choice_from_column(table, "管理人"),
        year=year or choice(years),
        compare=compare,
        percent=percent or randint(1, 9) / 10,
        sign=compare_to_sign[compare]
    )


@Manager(
    cluster=10,
    question_template="{date}港股{compare}的股票家数有多少家?",
    # id=295('20201211港股下跌的股票家数有多少家?')的答案是2961，即使用 昨收盘、不使用DISTINCT
    sql_template="""
    SELECT COUNT(股票代码) AS 数量
    FROM 港股票日行情表
    WHERE 交易日 = '{date}'
    AND [收盘价(元)] {sign} [昨收盘(元)]
    LIMIT 1;
    """
)
def gen10(date=None, compare=None):
    compare_to_sign = {
        "上涨": ">",
        "下跌": "<"
    }
    compare = compare or choice_from_dict(compare_to_sign)
    table = "港股票日行情表"

    return dict(
        date=date or choice_from_column(table, "交易日"),
        compare=compare,
        sign=compare_to_sign[compare]
    )


@Manager(
    cluster=11,
    question_template="我想了解{name}基金,在{year}年{season}的季报第{rank}大重股。该持仓股票当个季度的涨跌幅?请四舍五入保留百分比到小数点两位。",
    # 先把股票找出来，存到t1表中（只有一条数据），股票可能在A股表也可能在港股表，SQL貌似不支持动态选择表的操作，因此只能分别查询A股和港股表，
    # 然后将答案union起来得到t4表（有且只有1个表有数据）。t4表包含目标股票在目标季度内的所有数据，分别找到最早的昨收价和最晚的收盘价（不一定在季度第一天和最后一天），
    # 存到t5、t6表，最后就能算季度的涨跌幅了（join左右子表都只有1条数据，因此不用写on）
    sql_template="""
    WITH t1 AS (
        SELECT 股票代码
        FROM 基金股票持仓明细
        WHERE 基金简称 = '{name}'
        AND 持仓日期 = '{year}{end}'
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
        SELECT [昨收盘(元)] AS 期初价
        FROM t4
        ORDER BY 交易日 ASC
        LIMIT 1
    ),
    t6 AS (
        SELECT [收盘价(元)] AS 期末价
        FROM t4
        ORDER BY 交易日 DESC
        LIMIT 1
    )
    SELECT ROUND((t6.期末价 / t5.期初价 - 1) * 100, 2) || '%' AS 涨跌幅
    FROM t5 JOIN t6
    LIMIT 1;
    """
)
def gen11(name=None, year=None, season=None, rank=None):
    season_to_start_end = {
        "一季度": ("0101", "0331"),
        "二季度": ("0401", "0630"),
        "三季度": ("0701", "0930"),
        "四季度": ("1001", "1231")
    }
    season = season or choice_from_dict(season_to_start_end)
    start, end = season_to_start_end[season]
    table = "基金股票持仓明细"

    return dict(
        name=name or choice_from_column(table, "基金简称"),
        year=year or choice(years),
        season=season,
        rank=rank or randint(1, 9),
        start=start,
        end=end
    )


@Manager(
    cluster=12,
    question_template="我想知道{name}基金，在{year}年{report}中，前{rank}大重仓股中，有多少只股票在报告期内取得{compare}收益。",
    # 问题11的加强版，从计算一个股票改为计算多个股票
    # t5、t6还可以用 RANK() OVER (PARTITION BY 股票代码 ORDER BY 交易日 ASC) 的方式排序然后选择第1个，更通用
    # 由于t5、t6有“GROUP BY 股票代码”，因此最后的股票代码一定是唯一的，不加DISTINCT
    sql_template="""
    WITH t1 AS (
        SELECT 股票代码
        FROM 基金股票持仓明细
        WHERE 基金简称 = '{name}'
        AND 持仓日期 = '{year}{end}'
        AND 报告类型 = '年报(含半年报)'
        AND 第N大重仓股 <= {rank}
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
        SELECT 股票代码, [昨收盘(元)] AS 期初价
        FROM t4
        GROUP BY 股票代码
        HAVING 交易日 = MIN(交易日)
    ),
    t6 AS (
        SELECT 股票代码, [收盘价(元)] AS 期末价
        FROM t4
        GROUP BY 股票代码
        HAVING 交易日 = MAX(交易日)
    )
    SELECT COUNT(*) AS 数量
    FROM t5 JOIN t6
    ON t5.股票代码 = t6.股票代码
    WHERE t6.期末价 {sign} t5.期初价
    LIMIT 1;
    """
)
def gen12(name=None, year=None, report=None, rank=None, compare=None):
    report_to_start_end = {
        "半年度报告": ("0101", "0630"),
        "年度报告": ("0101", "1231")
    }
    report = report or choice_from_dict(report_to_start_end)
    start, end = report_to_start_end[report]

    compare_to_sign = {
        "正": ">",
        "负": "<"
    }
    compare = compare or choice_from_dict(compare_to_sign)
    table = "基金股票持仓明细"

    return dict(
        name=name or choice_from_column(table, "基金简称"),
        year=year or choice(years),
        report=report,
        rank=rank or randint(1, 10),
        compare=compare,
        start=start,
        end=end,
        sign=compare_to_sign[compare]
    )


@Manager(
    cluster=13,
    question_template="{date}日，{target}最大的前{rankzh}家上市公司的股票代码是什么？按成交金额从大到小给出",
    # A股和港股都有数据，不确定问题在问哪个，决定两个都查然后UNION起来。评价指标以召回率为主，这样做ok
    # 但发现港股有不少同一股票使用不同的股票代码，除了代码以外，其它字段完全一样，导致港股前三往往实际是同一只股票
    sql_template="""
    WITH t1 AS (
        SELECT 'A股' AS 市场, 股票代码, [{column}]
        FROM A股票日行情表
        WHERE 交易日 = '{date}'
        ORDER BY [{column}] DESC
        LIMIT {rank}
    ),
    t2 AS (
        SELECT '港股' AS 市场, 股票代码, [{column}]
        FROM 港股票日行情表
        WHERE 交易日 = '{date}'
        ORDER BY [{column}] DESC
        LIMIT {rank}
    )
    SELECT 市场, 股票代码
    FROM (
        SELECT * FROM t1
        UNION
        SELECT * FROM t2
    )
    ORDER BY 市场 ASC, [{column}] DESC;
    """
)
def gen13(date=None, target=None, rankzh=None):
    target = target or choice_from_dict(target_to_column)
    rankzh = rankzh or choice_from_dict(rankzh_to_rank)
    table = "A股票日行情表"

    return dict(
        date=date or choice_from_column(table, "交易日"),
        target=target,
        rankzh=rankzh,
        column=target_to_column[target],
        rank=rankzh_to_rank[rankzh]
    )


@Manager(
    cluster=14,
    question_template="{date}日，请给出{name}基金的管理人和累计单位净值。",
    sql_template="""
    SELECT t1.管理人, t2.累计单位净值
    FROM 基金基本信息 t1 JOIN 基金日行情表 t2
    ON t1.基金代码 = t2.基金代码
    AND t1.基金简称 = '{name}'
    AND t2.交易日期 = '{date}'
    LIMIT 1;
    """
)
def gen14a(date=None, name=None):
    table1 = "基金基本信息"
    table2 = "基金日行情表"

    return dict(
        date=date or choice_from_column(table2, "交易日期"),
        name=name or choice_from_column(table1, "基金简称")
    )


@Manager(
    cluster=14,
    question_template="{date}日，请给出{name}基金的管理人和单位净值。单位净值保留两位小数。",
    sql_template="""
    SELECT t1.管理人, ROUND(t2.单位净值, 2) AS 单位净值
    FROM 基金基本信息 t1 JOIN 基金日行情表 t2
    ON t1.基金代码 = t2.基金代码
    AND t1.基金简称 = '{name}'
    AND t2.交易日期 = '{date}'
    LIMIT 1;
    """
)
def gen14b(date=None, name=None):
    table1 = "基金基本信息"
    table2 = "基金日行情表"

    return dict(
        date=date or choice_from_column(table2, "交易日期"),
        name=name or choice_from_column(table1, "基金简称")
    )


@Manager(
    cluster=15,
    question_template="请列出{manager}在{year}年成立并且托管人为{trustee}的所有基金的基金{column}的平均数。",
    sql_template="""
    SELECT AVG({column}) AS 平均{column}
    FROM 基金基本信息
    WHERE 管理人 = '{manager}'
    AND 托管人 = '{trustee}'
    AND 成立日期 LIKE '{year}%' 
    LIMIT 1;
    """
)
def gen15(manager=None, year=None, trustee=None, column=None):
    columns = ["管理费率", "托管费率"]
    table = "基金基本信息"

    return dict(
        manager=manager or choice_from_column(table, "管理人"),
        year=year or choice(years),
        trustee=trustee or choice_from_column(table, "托管人"),
        column=column or choice(columns)
    )
