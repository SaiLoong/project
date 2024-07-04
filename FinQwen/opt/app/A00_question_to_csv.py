# -*- coding: utf-8 -*-
# @file A00_question_to_csv.py
# @author zhangshilong
# @date 2024/7/4

import csv

import jsonlines

QUESTION_NUM = 1000
WORKSPACE_DIR = "/mnt/workspace"
DATASET_DIR = f"{WORKSPACE_DIR}/bs_challenge_financial_14b_dataset"
INTERMEDIATE_DIR = f"{WORKSPACE_DIR}/intermediate"


def read_jsonl(path):
    with jsonlines.open(path, "r") as f:
        return list(f.iter(type=dict, skip_invalid=True))


def write_jsonl(path, data):
    with jsonlines.open(path, "w") as f:
        f.write_all(data)


data = read_jsonl(f"{DATASET_DIR}/question.json")
assert len(data) == QUESTION_NUM

with open(f"{INTERMEDIATE_DIR}/question_csv.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["问题id", "问题"])
    for item in data:
        writer.writerow([item["id"], item["question"]])
