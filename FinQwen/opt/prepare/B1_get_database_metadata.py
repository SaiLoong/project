# -*- coding: utf-8 -*-
# @file B1_get_database_metadata.py
# @author zhangshilong
# @date 2024/7/12

import pandas as pd
from tqdm import tqdm

from ..tools.config import Config
from ..tools.utils import File

db = Config.get_database()

# 获取所有表信息
tables = db.query("SELECT name FROM sqlite_master WHERE type='table';")["name"].tolist()
"""
["基金基本信息", "基金股票持仓明细", "基金债券持仓明细", "基金可转债持仓明细", "基金日行情表",
"A股票日行情表", "港股票日行情表", "A股公司行业划分表", "基金规模变动表", "基金份额持有人结构"]
"""

# 每个表抽样1000条保存
sample_database_dir = f"{Config.PREPARE_OUTPUT_DIR}/sample_database"
File.makedirs(sample_database_dir)
for table in tqdm(tables):
    # 貌似结果是固定的
    sample_df = db.query(f"SELECT * FROM {table} LIMIT {Config.DATABASE_RECORD_SAMPLE_NUM};")
    File.dataframe_to_csv(sample_df, f"{sample_database_dir}/{table}.csv")

# 收集每个表每个字段的唯一值信息
db_metadata = dict()
for table in tqdm(tables):  # 2min
    def func(row):
        column = row["字段名"]

        # 字段名中可能含有括号，因此用[]包起来（单引号、双引号其实也可以）
        distinct_num = db.query(f"SELECT COUNT(DISTINCT([{column}])) FROM {table};").iloc[0, 0]
        # 不随机的话，股票/基金代码等字段的值很小
        distinct_values = db.query(
            f"SELECT DISTINCT([{column}]) FROM {table} ORDER BY RANDOM() LIMIT {Config.COLUMN_DISTINCT_VALUE_SAMPLE_NUM};"
        )[column].tolist()

        return pd.Series({"唯一值数量": distinct_num, "唯一值抽样": distinct_values})


    df1 = db.query(f"PRAGMA table_info({table});")  # 获取表有哪些字段
    df2 = pd.DataFrame({"字段名": df1["name"], "字段类型": df1["type"]})
    column_df = pd.concat([df2, df2.apply(func, axis=1)], axis=1)
    # column_df.replace(np.nan, None, inplace=True)  # np.nan在json dump时直接生成NaN，不合法，因此转为None

    db_metadata[table] = {
        # 原本是np.int64，json dump会失败
        "记录数量": int(db.query(f"SELECT COUNT(*) FROM {table};").iloc[0, 0]),
        "字段信息": column_df.to_dict(orient="records")
    }

# 保存db_metadata
File.json_dump(db_metadata, Config.DATABASE_METADATA_PATH)
