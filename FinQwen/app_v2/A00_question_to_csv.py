import csv

import jsonlines

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
# TODO 原本用'utf-8-sig'
with open(f"{INTERMEDIATE_DIR}/question_csv.csv", "w", newline="", encoding="utf-8") as f:
    csvwriter = csv.writer(f)
    csvwriter.writerow(["问题id", "问题"])
    for item in data:
        # TODO 原版废代码：temp_question = temp_question.replace(' ', '')
        csvwriter.writerow([item["id"], item["question"]])
