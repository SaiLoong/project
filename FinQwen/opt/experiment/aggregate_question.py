# -*- coding: utf-8 -*-
# @file aggregate_question.py
# @author zhangshilong
# @date 2024/7/8

from collections import Counter

import pandas as pd
from sklearn.cluster import SpectralClustering
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import pairwise_distances

from ..tools.config import Config
from ..tools.constant import Category

tokenizer = Config.get_tokenizer()

question_category_df = Config.get_question_category_df()
question_df = question_category_df.query(f"分类 == '{Category.SQL}'").reset_index()[["问题id", "问题"]]

questions = question_df["问题"].tolist()

# TODO 看是否需要过滤常用词，根据检验结果判断
batch_token_ids = tokenizer(questions)["input_ids"]
counters = [Counter(tokens) for tokens in batch_token_ids]
vectors = DictVectorizer(sparse=False).fit_transform(counters)
distance_matrix = pairwise_distances(vectors, metric="jaccard")

# TODO 原版聚75类，是否是最优？  用法写进笔记
question_df["问题聚类"] = SpectralClustering(
    n_clusters=75,
    affinity="precomputed_nearest_neighbors",
    # n_neighbors=10,
    # verbose=True
).fit_predict(distance_matrix)

# TODO 展示结果、校验结果
for label, df in question_df.groupby("问题聚类"):
    print(f"聚类类别：{label}")
    for question in df["问题"]:
        print(f"\t{question}")
    print()

# TODO 测试集增加奇怪的缩写，prompt时告诉LLM有哪些公司
"""
生态环境建设行业的上游是什么行业？
"""

# TODO ref 一直卡在874（875），但CPU拉满，可能是sql有问题

# ======================================

pd.DataFrame.groupby

data0 = [
    "昨天天气真好！今天天气会怎么样呢？",
    "昨天天气很差，但是那场电影《天气之子》更差！",
    "如果明天天气不错，就去电影院看电影，否则回家看电影"
]

data = [
    "请问股票代码为002641的股票在2021年内日成交量低于该股票当年平均日成交量的有多少个交易日？",
    "请问股票代码为002771的股票在2021年内日成交量低于该股票当年平均日成交量的有多少个交易日？",

    "请查询在2019年度，603789股票涨停天数？   解释：（收盘价/昨日收盘价-1）》=9.8% 视作涨停",
    "请查询在2019年度，002093股票涨停天数？   解释：（收盘价/昨日收盘价-1）》=9.8% 视作涨停",

    "我想知道在2020年，广发基金管理有限公司已发行的基金中，有多少只基金报告期期初基金总份额小于报告期期末基金总份额(使用每只基金当年最晚的定期报告数据计算)？",
    "我想知道在2021年，格林基金管理有限公司已发行的基金中，有多少只基金报告期期初基金总份额小于报告期期末基金总份额(使用每只基金当年最晚的定期报告数据计算)？",

    "请帮我查询在截止2021-09-30的报告期间，基金总份额降低的基金数量是多少？",
    "请帮我查询在截止2019-05-31的报告期间，基金总份额降低的基金数量是多少？"
]

batch_token_ids = tokenizer(data)["input_ids"]

counters = [Counter(tokens) for tokens in batch_token_ids]

vectors = DictVectorizer(sparse=False).fit_transform(counters)

distance_matrix = pairwise_distances(vectors, metric="jaccard")

# TODO
# assign_labels{‘kmeans’, ‘discretize’, ‘cluster_qr’}, default=’kmeans’
# n_neighbors貌似是用来连接构成图的！
labels = SpectralClustering(n_clusters=4, affinity="precomputed_nearest_neighbors",
                            n_neighbors=4,
                            verbose=True).fit_predict(distance_matrix)
