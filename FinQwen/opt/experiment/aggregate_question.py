# -*- coding: utf-8 -*-
# @file aggregate_question.py
# @author zhangshilong
# @date 2024/7/8

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import SpectralClustering
from sklearn.metrics import silhouette_score
from tqdm import tqdm

from ..tools.config import Config
from ..tools.constant import Category

tokenizer = Config.get_tokenizer()

question_category_df = Config.get_question_classification_df()
question_df = question_category_df.query(f"问题分类 == '{Category.SQL}'").reset_index()[["问题id", "问题"]]  # 600条
questions = question_df["问题"].tolist()

distance_matrix = tokenizer.pairwise_jaccard_distances(questions)
score_matrix = 1 - distance_matrix

# 使用谱聚类，设置不同的簇数，通过轮廓系数衡量效果，画图并选出最优一个
cluster_nums = list(range(50, 70 + 1))
labels_list = list()
eval_scores = list()
for cluster_num in tqdm(cluster_nums):
    # 谱聚类会使用k-means, 其初始化存在随机性
    labels = SpectralClustering(n_clusters=cluster_num, affinity="precomputed", random_state=Config.SEED
                                ).fit_predict(score_matrix)
    labels_list.append(labels)
    eval_score = silhouette_score(distance_matrix, labels, metric="precomputed")
    eval_scores.append(eval_score)

sns.lineplot(x=cluster_nums, y=eval_scores)
plt.title("聚类效果")
plt.xlabel("簇数量")
plt.ylabel("轮廓系数")
plt.show()

best_idx = np.argmax(eval_scores)
best_cluster_num = cluster_nums[best_idx]  # 57
best_eval_score = eval_scores[best_idx]  # 0.6397
best_labels = labels_list[best_idx]

# 展示聚类结果
question_df["问题聚类标签"] = best_labels
for label, df in question_df.groupby("问题聚类标签"):
    print(f"聚类标签：{label}")
    for question in df["问题"]:
        print(f"\t{question}")
    print()

"""
结论：簇数为57时轮廓系数最大，0.6411

如果不设置随机种子，最优簇数在55-65之间徘徊
"""
# TODO 逐个簇处理SQL、是否需要使用停用词
