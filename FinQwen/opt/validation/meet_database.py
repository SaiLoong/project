# -*- coding: utf-8 -*-
# @file meet_database.py
# @author zhangshilong
# @date 2024/7/5

import csv
import sqlite3

WORKSPACE_DIR = "/mnt/workspace"
DATASET_DIR = f"{WORKSPACE_DIR}/bs_challenge_financial_14b_dataset"
VALIDATION_DIR = f"{WORKSPACE_DIR}/validation"
DATABASE_DIR = f"{VALIDATION_DIR}/database"

TABLE_NAMES = ["基金基本信息", "基金股票持仓明细", "基金债券持仓明细", "基金可转债持仓明细", "基金日行情表",
               "A股票日行情表", "港股票日行情表", "A股公司行业划分表", "基金规模变动表", "基金份额持有人结构"]
SAMPLE_NUM = 1000

conn = sqlite3.connect(f"{DATASET_DIR}/dataset/博金杯比赛数据.db")
cursor = conn.cursor()

# 查看每个表的数量
for table_name in TABLE_NAMES:
    cursor.execute(f"select count(*) from {table_name};")
    count = cursor.fetchall()[0][0]
    print(f"{table_name=} {count=}")

# 每个表取1000条保存下来
for table_name in TABLE_NAMES:
    with open(f"{DATABASE_DIR}/{table_name}_sample{SAMPLE_NUM}.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        cursor.execute(f"select * from {table_name} limit {SAMPLE_NUM};")
        writer.writerow([description[0] for description in cursor.description])
        writer.writerows(cursor.fetchall())
