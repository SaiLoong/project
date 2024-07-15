# -*- coding: utf-8 -*-
# @file sql_generator_v2.py
# @author zhangshilong
# @date 2024/7/15

from dataclasses import dataclass
from random import choice

from sql_utils_v2 import Generator
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
    answer_template: str = "在{year}年{month}月的报告中，报告期基金总申购份额和报告期基金总赎回份额差额最大的一只基金的简称是{基金简称}，差额是{差额:.2f}份"

    def preprocess_params(self, year=None, month=None):
        months = ["03", "06", "09", "12"]

        return dict(
            year=year or choice(years),
            month=month or choice(months)
        )
