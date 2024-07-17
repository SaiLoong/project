# -*- coding: utf-8 -*-
# @file sql_generator_v2.py
# @author zhangshilong
# @date 2024/7/15

import re
from dataclasses import dataclass
from random import choice
from random import randint

from sql_utils import Generator
from ..tools.config import Config

# TODO debug
db = Config.get_database()

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
shortyears = [year - 2000 for year in years]
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


@dataclass
class Generator0(Generator):
    cluster: int = 0
    question_template: str = "请帮我查询下，在{year}年{month}月的报告中，报告期基金总申购份额和报告期基金总赎回份额差额最大的一只基金的简称是什么？差额有多少？保留两位小数。"
    sql_template: str = """
    SELECT 基金简称, ROUND(报告期基金总申购份额 - 报告期基金总赎回份额, 2) AS 差额
    FROM 基金规模变动表
    WHERE 截止日期 LIKE '{year}-{month}%'
    ORDER BY 差额 DESC
    LIMIT 1;
    """
    answer_template: str = "在{year}年{month}月的报告中，报告期基金总申购份额和报告期基金总赎回份额差额最大的一只基金的简称是{基金简称}，差额是{差额:.2f}份。"
    verification_score: float = 1.15  # 满分1.17

    def preprocess_params(self, year=None, month=None):
        months = ["03", "06", "09", "12"]

        return dict(
            year=year or choice(years),
            month=month or choice(months)
        )


@dataclass
class Generator1(Generator):
    cluster: int = 1
    question_template: str = "请帮我查询在截止{year}-{monthday}的基金定期报告中，{target}为零的基金有几个？"
    sql_template: str = """
    SELECT COUNT(基金代码) AS 数量
    FROM 基金规模变动表
    WHERE 截止日期 LIKE '{year}-{monthday}%'
    AND 报告类型 = '基金定期报告'
    AND 报告期{target} = 0
    LIMIT 1;
    """
    answer_template: str = "在截止{year}-{monthday}的基金定期报告中，{target}为零的基金有{数量}个。"
    verification_score: float = 1.33  # 满分1.33

    def preprocess_params(self, year=None, monthday=None, target=None):
        monthdays = ["03-31", "06-30", "09-30", "12-31"]
        targets = ["期初基金总份额", "基金总申购份额", "基金总赎回份额", "期末基金总份额"]

        return dict(
            year=year or choice(years),
            monthday=monthday or choice(monthdays),
            target=target or choice(targets)
        )


@dataclass
class Generator2(Generator):
    cluster: int = 2
    question_template: str = "帮我查一下在{year}年，代码为{code}的{stock}股票今开盘{compare}昨收盘的天数？"
    sql_template: str = """
    SELECT COUNT(交易日) AS 天数
    FROM {table}
    WHERE 股票代码 = '{code}'
    AND 交易日 LIKE '{year}%'
    AND [今开盘(元)] {sign} [昨收盘(元)]
    LIMIT 1;  
    """
    answer_template: str = "在{year}年，代码为{code}的{stock}股票今开盘{compare}昨收盘的天数是{天数}天。"
    verification_score: float = 1.81  # 满分1.83

    def preprocess_params(self, year=None, code=None, stock=None, compare=None):
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


@dataclass
class Generator3a(Generator):
    cluster: int = 3
    question_template: str = "针对{year}年的{report}，有多少家基金的{role}持有份额占比不足{percent}%?"
    sql_template: str = """
    SELECT COUNT(基金代码) AS 数量
    FROM 基金份额持有人结构
    WHERE 定期报告所属年度 = {year}
    AND 报告类型 = '{report}'
    AND {role}持有的基金份额占总份额比例 < {percent}
    LIMIT 1;
    """
    answer_template: str = "针对{year}年的{report}，有{数量}家基金的{role}持有份额占比不足{percent}%。"
    verification_score: float = 0.32  # 满分0.33

    def preprocess_params(self, year=None, report=None, role=None, percent=None):
        roles = ["个人投资者", "机构投资者"]
        table = "基金份额持有人结构"

        return dict(
            year=year or choice(years),
            report=report or choice_from_column(table, "报告类型"),
            role=role or choice(roles),
            percent=percent or randint(1, 100)
        )


@dataclass
class Generator3b(Generator):
    cluster: int = 3
    question_template: str = "请帮我查询下，在{year}年{season}季报报告中，{code}基金的第{rankzh}大重仓可转债同期还有多少只基金也进行了持仓？"
    # “WHERE 对应股票代码 = (..)”里共限定了四个条件，其实已经保证必然只有一条记录。外面有三个条件限制，因此 基金代码 必然唯一
    # 问题是问“还有”，因此要扣除自己
    sql_template: str = """
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
    answer_template: str = "在{year}年{season}季报报告中，{code}基金的第{rankzh}大重仓可转债同期还有{数量}只基金也进行了持仓。"
    verification_score: float = 0.33  # 满分0.33

    def preprocess_params(self, year=None, season=None, code=None, rankzh=None):
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


@dataclass
class Generator4(Generator):
    cluster: int = 4
    question_template: str = "我想知道{name}基金在{date}的{report}中，其可转债持仓占比最大的是哪个行业？用{standard}一级行业来统计。"
    sql_template: str = """
    WITH t1 AS (
        SELECT 对应股票代码, 市值
        FROM 基金可转债持仓明细
        WHERE 基金简称 = '{name}'
        AND 持仓日期 = '{date}'
        AND 报告类型 = '{report}'
    ),
    t2 AS (
        SELECT 股票代码, 一级行业名称
        FROM A股公司行业划分表
        WHERE 行业划分标准 = '{standard}行业分类'
        AND 交易日期 = '{date}'
    )
    SELECT 一级行业名称
    FROM t1 JOIN t2
    ON t1.对应股票代码 = t2.股票代码
    GROUP BY 一级行业名称
    ORDER BY SUM(市值) DESC
    LIMIT 1;
    """
    answer_template: str = "{name}基金在{date}的{report}中，其可转债持仓占比最大的是{standard}一级行业划分标准下的{一级行业名称}行业。"
    verification_score: float = 4.38  # 满分4.5，应该是id=326数据有问题

    def preprocess_params(self, name=None, date=None, report=None, standard=None):
        table = "基金可转债持仓明细"

        return dict(
            name=name or choice_from_column(table, "基金简称"),
            date=date or choice_from_column(table, "持仓日期"),
            report=report or choice_from_column(table, "报告类型"),
            standard=standard or choice(standards)
        )


@dataclass
class Generator5(Generator):
    cluster: int = 5
    question_template: str = "在{date}的{report}中，{name}基金的债券持仓,其持有最大仓位的债券类型是什么?"
    # 一开始直接取”第N大重仓股“最小的那个，分数只有1.77
    sql_template: str = """
    SELECT 债券类型
    FROM 基金债券持仓明细
    WHERE 持仓日期 = '{date}'
    AND 报告类型 = '{report}'
    AND 基金简称 = '{name}'
    GROUP BY 债券类型
    ORDER BY SUM(持债市值) DESC
    LIMIT 1;
    """

    answer_template: str = "在{date}的{report}中，{name}基金的债券持仓,其持有最大仓位的债券类型是{债券类型}。"
    verification_score: float = 1.97  # 满分2.0

    def preprocess_params(self, date=None, report=None, name=None):
        table = "基金债券持仓明细"

        return dict(
            date=date or choice_from_column(table, "持仓日期"),
            report=report or choice_from_column(table, "报告类型"),
            name=name or choice_from_column(table, "基金简称")
        )


@dataclass
class Generator6(Generator):
    cluster: int = 6
    question_template: str = "{manager}管理的{category}产品的数量有多少?"
    sql_template: str = """
    SELECT COUNT(基金代码) AS 数量
    FROM 基金基本信息
    WHERE 管理人 = '{manager}'
    AND 基金类型 = '{category}'
    LIMIT 1;
    """
    answer_template: str = "{manager}管理的{category}产品的数量{数量}个。"
    verification_score: float = 0.80  # 满分0.83

    def preprocess_params(self, manager=None, category=None):
        table = "基金基本信息"

        return dict(
            manager=manager or choice_from_column(table, "管理人"),
            category=category or choice_from_column(table, "基金类型")
        )


@dataclass
class Generator7a(Generator):
    cluster: int = 7
    question_template: str = "{date}日，一级行业为{industry1}的股票的{target}合计是多少？取整。"
    # 问题没有明确是中信还是申万标准，有的一级行业只有一边有，有的两边都有
    # A股票日行情表 里，股票代码 + 交易日 是唯一的
    sql_template: str = """
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
    answer_template: str = "{date}日，一级行业为{industry1}的股票的{target}合计是{result}。"
    verification_score: float = 2.92  # 满分3.17。可能是中信/申万标准的问题

    def preprocess_params(self, date=None, industry1=None, target=None):
        target_to_column = {
            "成交量": "成交量(股)",
            "成交金额": "成交金额(元)"
        }
        target = target or choice_from_dict(target_to_column)
        table = "A股公司行业划分表"

        return dict(
            date=date or choice_from_column(table, "交易日期"),
            industry1=industry1 or choice_from_column(table, "一级行业名称"),
            target=target,
            column=target_to_column[target]
        )

    def postprocess_result(self, result, params):
        # SUM一定返回一行记录，但值有可能是None
        target, unit = re.fullmatch(r"(.*)\((.*)\)", params["column"]).groups()
        result = result[0][f"{target}合计"]
        return None if result is None else dict(result=str(result) + unit)


@dataclass
class Generator7b(Generator):
    cluster: int = 7
    question_template: str = "请帮我计算：在{date}，日收益率为{compare}的{stock}股票有几个。"
    sql_template: str = """
    SELECT COUNT(股票代码) AS 数量
    FROM {table}
    WHERE 交易日 = '{date}'
    AND [收盘价(元)] {sign} [昨收盘(元)]
    LIMIT 1;
    """
    answer_template: str = "在{date}，日收益率为{compare}的{stock}股票有{数量}个。"
    verification_score: float = 0.17  # 满分0.17

    def preprocess_params(self, date=None, compare=None, stock=None):
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


@dataclass
class Generator8(Generator):
    cluster: int = 8
    question_template: str = "请帮我计算，在{date}，{standard}行业分类划分的一级行业为{industry1}行业中，涨跌幅最大股票的股票代码是？涨跌幅是多少？百分数保留两位小数。股票涨跌幅定义为：（收盘价 - 前一日收盘价 / 前一日收盘价）* 100%。"
    sql_template: str = """
    SELECT 股票代码, ROUND(涨跌幅0 * 100, 2) AS 涨跌幅
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
    answer_template: str = "在{date}，{standard}行业分类划分的一级行业为{industry1}行业中，涨跌幅最大股票的股票代码是{股票代码}，涨跌幅是{涨跌幅:.2f}%。"
    verification_score: float = 1.66  # 满分1.67

    def preprocess_params(self, date=None, standard=None, industry1=None):
        table = "A股公司行业划分表"

        return dict(
            date=date or choice_from_column(table, "交易日期"),
            standard=standard or choice(standards),
            industry1=industry1 or choice_from_column(table, "一级行业名称")
        )


@dataclass
class Generator9(Generator):
    cluster: int = 9
    question_template: str = "我想知道{company}在{year}年成立了多少只管理费率{compare}于{percent}%的基金？"
    sql_template: str = """
    SELECT COUNT(基金代码) AS 数量
    FROM 基金基本信息
    WHERE 管理人 = '{company}'
    AND 成立日期 LIKE '{year}%'
    AND 管理费率 {sign} '{percent}%'
    LIMIT 1;
    """
    answer_template: str = "{company}在{year}年成立了{数量}只管理费率{compare}于{percent}%的基金。"
    verification_score: float = 1.5  # 满分1.5

    def preprocess_params(self, company=None, year=None, compare=None, percent=None):
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


@dataclass
class Generator10(Generator):
    cluster: int = 10
    question_template: str = "{date}港股{compare}的股票家数有多少家?"
    sql_template: str = """
    SELECT COUNT(股票代码) AS 数量
    FROM 港股票日行情表
    WHERE 交易日 = '{date}'
    AND [收盘价(元)] {sign} [昨收盘(元)]
    LIMIT 1;
    """
    answer_template: str = "{date}港股{compare}的股票家数有{数量}家。"
    verification_score: float = 1.65  # 满分1.67

    def preprocess_params(self, date=None, compare=None):
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


@dataclass
class Generator11(Generator):
    cluster: int = 11
    question_template: str = "我想了解{name}基金,在{year}年{season}的季报第{rank}大重股。该持仓股票当个季度的涨跌幅?请四舍五入保留百分比到小数点两位。"
    # t1表包含目标股票在目标季度内的所有数据，分别找到最早的昨收价和最晚的收盘价（不一定在季度第一天和最后一天）存到t2、t3表
    # 最后就能算季度的涨跌幅了（join左右子表都只有1条数据，因此不用写on）
    sql_template: str = """
    WITH t1 AS (
        SELECT *
        FROM A股票日行情表
        WHERE 股票代码 = (
            SELECT 股票代码
            FROM 基金股票持仓明细
            WHERE 基金简称 = '{name}'
            AND 持仓日期 = '{year}{end}'
            AND 报告类型 = '季报'
            AND 第N大重仓股 = {rank}
            LIMIT 1
        )
        AND 交易日 BETWEEN '{year}{start}' AND '{year}{end}'
    ),
    t2 AS (
        SELECT [昨收盘(元)] AS 期初价
        FROM t1
        ORDER BY 交易日 ASC
        LIMIT 1
    ),
    t3 AS (
        SELECT [收盘价(元)] AS 期末价
        FROM t1
        ORDER BY 交易日 DESC
        LIMIT 1
    )
    SELECT ROUND((t3.期末价 / t2.期初价 - 1) * 100, 2) AS 涨跌幅
    FROM t2 JOIN t3
    LIMIT 1;
    """
    answer_template: str = "{name}基金,在{year}年{season}的季报第{rank}大重股当个季度的涨跌幅是{涨跌幅:.2f}%。"
    verification_score: float = 1.55  # 满分1.67，纯答案1.27，看不出缺少的分数是语义不接近还是错了1题。加不加港股的答案都是一样的

    def preprocess_params(self, name=None, year=None, season=None, rank=None):
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


@dataclass
class Generator12(Generator):
    cluster: int = 12
    question_template: str = "我想知道{name}基金，在{year}年{report}中，前{rank}大重仓股中，有多少只股票在报告期内取得{compare}收益。"
    # 问题11的加强版，从计算一个股票改为计算多个股票
    # t2、t3还可以用 RANK() OVER (PARTITION BY 股票代码 ORDER BY 交易日 ASC) 的方式排序然后选择第1个，更通用
    # 由于t2、t3有“GROUP BY 股票代码”，因此最后的股票代码一定是唯一的，不加DISTINCT
    sql_template: str = """
    WITH t1 AS (
        SELECT *
        FROM A股票日行情表
        WHERE 股票代码 IN (
            SELECT 股票代码
            FROM 基金股票持仓明细
            WHERE 基金简称 = '{name}'
            AND 持仓日期 = '{year}{end}'
            AND 报告类型 = '年报(含半年报)'
            AND 第N大重仓股 <= {rank}
        )
        AND 交易日 BETWEEN '{year}{start}' AND '{year}{end}'
    ),
    t2 AS (
        SELECT 股票代码, [昨收盘(元)] AS 期初价
        FROM t1
        GROUP BY 股票代码
        HAVING 交易日 = MIN(交易日)
    ),
    t3 AS (
        SELECT 股票代码, [收盘价(元)] AS 期末价
        FROM t1
        GROUP BY 股票代码
        HAVING 交易日 = MAX(交易日)
    )
    SELECT COUNT(*) AS 数量
    FROM t2 JOIN t3
    ON t2.股票代码 = t3.股票代码
    WHERE t3.期末价 {sign} t2.期初价
    LIMIT 1;
    """
    answer_template: str = "{name}基金，在{year}年{report}中，前{rank}大重仓股中，有{数量}只股票在报告期内取得{compare}收益。"
    verification_score: float = 3.21  # 满分3.33。不能加港股，加了有6题答案变化，分数降至2.67

    def preprocess_params(self, name=None, year=None, report=None, rank=None, compare=None):
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


@dataclass
class Generator13(Generator):
    cluster: int = 13
    question_template: str = "{date}日，{target}最大的前{rankzh}家上市公司的股票代码是什么？按{target}从大到小给出"
    # 不考虑港股
    sql_template: str = """
    SELECT 股票代码
    FROM A股票日行情表
    WHERE 交易日 = '{date}'
    ORDER BY [{column}] DESC
    LIMIT {rank};
    """
    answer_template: str = "{date}日，{target}最大的前{rankzh}家上市公司的股票代码按{target}从大到小依次是{result}。"
    verification_score: float = 1.63  # 满分1.67

    def preprocess_params(self, date=None, target=None, rankzh=None):
        target_to_column = {
            "成交量": "成交量(股)",
            "成交金额": "成交金额(元)"
        }
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

    def postprocess_result(self, result, params):
        if result:
            return dict(
                result="、".join([record["股票代码"] for record in result])
            )
        return None


@dataclass
class Generator14a(Generator):
    cluster: int = 14
    question_template: str = "{date}日，请给出{name}基金的管理人和累计单位净值。"
    sql_template: str = """
    SELECT t1.管理人, t2.累计单位净值
    FROM 基金基本信息 t1 JOIN 基金日行情表 t2
    ON t1.基金代码 = t2.基金代码
    AND t1.基金简称 = '{name}'
    AND t2.交易日期 = '{date}'
    LIMIT 1;
    """
    answer_template: str = "{date}日，{name}基金的管理人是{管理人}，累计单位净值是{累计单位净值}元。"
    verification_score: float = 1.66  # 满分1.67

    def preprocess_params(self, date=None, name=None):
        table1 = "基金基本信息"
        table2 = "基金日行情表"

        return dict(
            date=date or choice_from_column(table2, "交易日期"),
            name=name or choice_from_column(table1, "基金简称")
        )


@dataclass
class Generator14b(Generator):
    cluster: int = 14
    question_template: str = "{date}日，请给出{name}基金的管理人和单位净值。单位净值保留两位小数。"
    sql_template: str = """
    SELECT t1.管理人, ROUND(t2.单位净值, 2) AS 单位净值
    FROM 基金基本信息 t1 JOIN 基金日行情表 t2
    ON t1.基金代码 = t2.基金代码
    AND t1.基金简称 = '{name}'
    AND t2.交易日期 = '{date}'
    LIMIT 1;
    """
    answer_template: str = "{date}日，{name}基金的管理人是{管理人}，单位净值是{单位净值:.2f}元。"
    verification_score: float = 0.50  # 满分0.5

    def preprocess_params(self, date=None, name=None):
        table1 = "基金基本信息"
        table2 = "基金日行情表"

        return dict(
            date=date or choice_from_column(table2, "交易日期"),
            name=name or choice_from_column(table1, "基金简称")
        )


@dataclass
class Generator15(Generator):
    cluster: int = 15
    question_template: str = "请列出{manager}在{year}年成立并且托管人为{trustee}的所有基金的基金{column}的平均数。"
    # 虽然 托管费率/管理费率 的值是形如"1.2%"的text，但是能应用AVG函数，视作将%删掉只保留前面的数值然后算平均值
    sql_template: str = """
    SELECT AVG({column}) AS 平均{column}
    FROM 基金基本信息
    WHERE 管理人 = '{manager}'
    AND 托管人 = '{trustee}'
    AND 成立日期 LIKE '{year}%'
    LIMIT 1;
    """
    answer_template: str = "{manager}在{year}年成立并且托管人为{trustee}的所有基金的基金{column}的平均数是{result}%。"
    verification_score: float = 1.65  # 满分1.67

    def preprocess_params(self, manager=None, year=None, trustee=None, column=None):
        columns = ["管理费率", "托管费率"]
        table = "基金基本信息"

        return dict(
            manager=manager or choice_from_column(table, "管理人"),
            year=year or choice(years),
            trustee=trustee or choice_from_column(table, "托管人"),
            column=column or choice(columns)
        )

    def postprocess_result(self, result, params):
        # AVG一定返回一行记录，但值有可能是None
        column = params["column"]
        result = result[0][f"平均{column}"]
        return None if result is None else dict(result=result)


@dataclass
class Generator16(Generator):
    cluster: int = 16
    question_template: str = "请查询：在{year}的{report}中，个人投资者持有基金份额{compare}机构投资者持有基金份额的基金属于{category}类型的有几个。"
    sql_template: str = """
    SELECT COUNT(*) AS 数量
    FROM 基金基本信息
    WHERE 基金代码 IN (
        SELECT 基金代码
        FROM 基金份额持有人结构
        WHERE 定期报告所属年度 = {year}
        AND 报告类型 = '{report}'
        AND 个人投资者持有的基金份额 {sign} 机构投资者持有的基金份额
    )
    AND 基金类型 = '{category}'
    LIMIT 1;
    """
    answer_template: str = "在{year}的{report}中，个人投资者持有基金份额{compare}机构投资者持有基金份额的基金属于{category}类型的有{数量}个。"
    verification_score: float = 2.00  # 满分2.0

    def preprocess_params(self, year=None, report=None, compare=None, category=None):
        compare_to_sign = {
            "大于": ">",
            "小于": "<"
        }
        compare = compare or choice_from_dict(compare_to_sign)
        table1 = "基金份额持有人结构"
        tabel2 = "基金基本信息"

        return dict(
            year=year or choice(years),
            report=report or choice_from_column(table1, "报告类型"),
            compare=compare,
            category=category or choice_from_column(tabel2, "基金类型"),
            sign=compare_to_sign[compare]
        )


@dataclass
class Generator17(Generator):
    cluster: int = 17
    question_template: str = "我想知道股票{code}在{standard}行业分类下的{level}行业是什么？用最新的数据。"
    sql_template: str = """
    SELECT {level}行业名称
    FROM A股公司行业划分表
    WHERE 股票代码 = '{code}'
    AND 行业划分标准 = '{standard}行业分类'
    ORDER BY 交易日期 DESC
    LIMIT 1;
    """
    answer_template: str = "股票{code}在{standard}行业分类下的{level}行业是{result}。"
    verification_score: float = 3.16  # 满分3.33，应该是语义分缺失

    def preprocess_params(self, code=None, standard=None, level=None):
        levels = ["一级", "二级"]
        table = "A股公司行业划分表"

        return dict(
            code=code or choice_from_column(table, "股票代码"),
            standard=standard or choice(standards),
            level=level or choice(levels)
        )

    def postprocess_result(self, result, params):
        if result:
            level = params["level"]
            return dict(
                result=result[0][f"{level}行业名称"]
            )
        return None


@dataclass
class Generator18(Generator):
    cluster: int = 18
    question_template: str = "{name}基金在{date}的{report}里，前{rankzh}大持仓占比的债券名称是什么?"
    sql_template: str = """
    SELECT 债券名称
    FROM 基金债券持仓明细
    WHERE 基金简称 = '{name}'
    AND 持仓日期 = '{date}'
    AND 报告类型 = '{report}'
    ORDER BY 第N大重仓股 ASC
    LIMIT {rank};
    """
    answer_template: str = "{name}基金在{date}的{report}里，前{rankzh}大持仓占比的债券名称是{result}。"
    verification_score: float = 3.27  # 满分3.33

    def preprocess_params(self, name=None, date=None, report=None, rankzh=None):
        rankzh = rankzh or choice_from_dict(rankzh_to_rank)
        table = "基金债券持仓明细"

        return dict(
            name=name or choice_from_column(table, "基金简称"),
            date=date or choice_from_column(table, "持仓日期"),
            report=report or choice_from_column(table, "报告类型"),
            rankzh=rankzh,
            rank=rankzh_to_rank[rankzh]
        )

    def postprocess_result(self, result, params):
        if result:
            return dict(
                result="、".join([record["债券名称"] for record in result])
            )
        return None


@dataclass
class Generator19(Generator):
    cluster: int = 19
    # 参数不能紧挨着，只能出此下策
    question_template: str = "{manager}20{shortyear}年成立了多少基金?"
    sql_template: str = """
    SELECT COUNT(*) AS 数量
    FROM 基金基本信息
    WHERE 管理人 = '{manager}'
    AND 成立日期 LIKE '20{shortyear}%'
    LIMIT 1;
    """
    answer_template: str = "{manager}20{shortyear}年成立了{数量}只基金。"
    verification_score: float = 1.64  # 满分1.67

    def preprocess_params(self, manager=None, shortyear=None):
        table = "基金基本信息"

        return dict(
            manager=manager or choice_from_column(table, "管理人"),
            shortyear=shortyear or choice(shortyears)
        )


@dataclass
class Generator20(Generator):
    cluster: int = 20
    question_template: str = "我想知道在{year}年，{manager}已发行的基金中，有多少只基金报告期期初基金总份额{compare}报告期期末基金总份额(使用每只基金当年最晚的定期报告数据计算)？"
    sql_template: str = """
    WITH t1 AS (
        SELECT *
        FROM 基金规模变动表
        WHERE 基金代码 IN (
            SELECT 基金代码
            FROM 基金基本信息
            WHERE 管理人 = '{manager}'
        )
        AND 定期报告所属年度 = {year}
        AND 报告类型 = '基金定期报告'
        GROUP BY 基金代码
        HAVING 截止日期 = MAX(截止日期)
    )
    SELECT COUNT(*) AS 数量
    FROM t1
    WHERE 报告期期初基金总份额 {sign} 报告期期末基金总份额
    LIMIT 1;
    """
    answer_template: str = "在{year}年，{manager}已发行的基金中，有{数量}只基金报告期期初基金总份额{compare}报告期期末基金总份额。"
    verification_score: float = 1.67  # 满分1.67

    def preprocess_params(self, year=None, manager=None, compare=None):
        compare_to_sign = {
            "大于": ">",
            "小于": "<"
        }
        compare = compare or choice_from_dict(compare_to_sign)
        table = "基金基本信息"

        return dict(
            year=year or choice(years),
            manager=manager or choice_from_column(table, "管理人"),
            compare=compare,
            sign=compare_to_sign[compare]
        )


@dataclass
class Generator21(Generator):
    cluster: int = 21
    question_template: str = "在{year}年的{report}里，{manager}管理的基金中，有多少比例的基金是个人投资者持有的份额{compare}机构投资者？希望得到一个精确到两位小数的百分比。"
    sql_template: str = """
    WITH t1 AS (
        SELECT *
        FROM 基金份额持有人结构
        WHERE 基金代码 IN (
            SELECT 基金代码
            FROM 基金基本信息
            WHERE 管理人 = '{manager}'
        )
        AND 定期报告所属年度 = {year}
        AND 报告类型 = '{report}'
    )
    SELECT ROUND(CAST(SUM(个人投资者持有的基金份额 {sign} 机构投资者持有的基金份额) AS FLOAT) / COUNT(*) * 100, 2) AS 比例
    FROM t1
    LIMIT 1;
    """
    answer_template: str = "在{year}年的{report}里，{manager}管理的基金中，有{result:.2f}%的基金是个人投资者持有的份额{compare}机构投资者。"
    # 将answer_template的{result:.2f}改为{result}，id=30/167/522/913的答案发生变化，分数降至0.90，说明这4题本来有1题答案就是错的
    # 分析发现id=522存在个人/机构的份额都为0的数据，以为这是原因，于是添加 个人投资者持有的基金份额 + 机构投资者持有的基金份额 > 0 的条件，
    #   id=522/673的答案发生变化，分数降至1.00，说明debug思路不对，而且这两题的答案本来是对的
    verification_score: float = 1.21  # 满分1.67，纯答案0.80，可能是有脏数据导致某些题查询结果不准确

    def preprocess_params(self, year=None, report=None, manager=None, compare=None):
        compare_to_sign = {
            "超过": ">",
            "不足": "<"
        }
        compare = compare or choice_from_dict(compare_to_sign)
        table1 = "基金份额持有人结构"
        table2 = "基金基本信息"

        return dict(
            year=year or choice(years),
            report=report or choice_from_column(table1, "报告类型"),
            manager=manager or choice_from_column(table2, "管理人"),
            compare=compare,
            sign=compare_to_sign[compare]
        )

    def postprocess_result(self, result, params):
        # SUM一定返回一行记录，但值有可能是None
        result = result[0]["比例"]
        return None if result is None else dict(result=result)


@dataclass
class Generator22(Generator):
    cluster: int = 22
    question_template: str = "{date}日，{target1}较上一交易日{target2}高的股票有多少只？（如上一交易日没有某只股票，则不统计在内）"
    # 交易日并不是连续的，不能直接对date-1
    # 如果用 GROUP BY 股票代码 HAVING 交易日=MAX(交易日)，无法保证所有股票的最后一天是同一天，与题意不符
    sql_template: str = """
    SELECT COUNT(*) AS 数量
    FROM A股票日行情表 t1 JOIN A股票日行情表 t2
    ON t1.股票代码 = t2.股票代码
    AND t1.交易日 = '{date}'
    AND t2.交易日 = (
        SELECT MAX(交易日)
        FROM A股票日行情表
        WHERE 交易日 < '{date}'
        LIMIT 1
    )
    AND t1.[{column1}] > t2.[{column2}]
    LIMIT 1;
    """
    answer_template: str = "{date}日，{target1}较上一交易日{target2}高的股票有{数量}只。"
    verification_score: float = 1.67  # 满分1.67

    def preprocess_params(self, date=None, target1=None, target2=None):
        target_to_column = {
            "开盘价": "今开盘(元)",
            "最高价": "最高价(元)",
            "最低价": "最低价(元)",
            "收盘价": "收盘价(元)"
        }
        target1 = target1 or choice_from_dict(target_to_column)
        target2 = target2 or choice_from_dict(target_to_column)
        table = "A股票日行情表"

        return dict(
            date=date or choice_from_column(table, "交易日"),
            target1=target1,
            target2=target2,
            column1=target_to_column[target1],
            column2=target_to_column[target2]
        )


@dataclass
class Generator23(Generator):
    cluster: int = 23
    question_template: str = "我想了解一下{manager}20{shortyear}年成立的{category}基金,其{column}的平均值是多少?请四舍五入保留小数点两位。"
    sql_template: str = """
    SELECT ROUND(AVG({column}), 2) AS 平均{column}
    FROM 基金基本信息
    WHERE 管理人 = '{manager}'
    AND 成立日期 LIKE '20{shortyear}%'
    AND 基金类型 = '{category}'
    LIMIT 1;
    """
    answer_template: str = "{manager}20{shortyear}年成立的{category}基金,其{column}的平均值是{result:.2f}%。"
    verification_score: float = 1.61  # 满分1.67

    def preprocess_params(self, manager=None, shortyear=None, category=None, column=None):
        columns = ["管理费率", "托管费率"]
        table = "基金基本信息"

        return dict(
            manager=manager or choice_from_column(table, "管理人"),
            shortyear=shortyear or choice(shortyears),
            category=category or choice_from_column(table, "基金类型"),
            column=column or choice(columns)
        )

    def postprocess_result(self, result, params):
        # AVG一定返回一行记录，但值有可能是None
        column = params["column"]
        result = result[0][f"平均{column}"]
        return None if result is None else dict(result=result)


# ====================================================================
# 模板

# TODO verify ing
@dataclass
class Generator(Generator):
    cluster: int = NotImplemented
    question_template: str = ""
    sql_template: str = ""
    answer_template: str = ""
    verification_score: float = None  # 满分

    def preprocess_params(self):
        table = ""

        return dict(
        )


question = ""

sql = """
SELECT *
FROM
WHERE
AND
AND

LIMIT 100;
"""

db.query(sql)
