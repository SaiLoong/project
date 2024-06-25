# -*- coding: utf-8 -*-
# @file finetune_prompt.py
# @author zhangshilong
# @date 2024/6/25

from typing import Dict

import torch
import transformers
from transformers import AutoTokenizer
from transformers.trainer_pt_utils import LabelSmoother

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def get_data():
    data = [
        {
            "id": "identity_0",
            "conversations": [
                {
                    "from": "user",
                    "value": "0+1=?"
                },
                {
                    "from": "assistant",
                    "value": "1"
                }
            ]
        }
    ]

    return data


def preprocess(
        sources,
        tokenizer: transformers.PreTrainedTokenizer,
        max_len: int,
        system_message: str = "You are a helpful assistant."
) -> Dict:
    roles = {"user": "<|im_start|>user", "assistant": "<|im_start|>assistant"}

    im_start = tokenizer.im_start_id
    im_end = tokenizer.im_end_id
    nl_tokens = tokenizer('\n').input_ids
    _system = tokenizer('system').input_ids + nl_tokens
    _user = tokenizer('user').input_ids + nl_tokens
    _assistant = tokenizer('assistant').input_ids + nl_tokens

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != roles["user"]:
            source = source[1:]

        input_id, target = [], []
        system = [im_start] + _system + tokenizer(system_message).input_ids + [im_end] + nl_tokens
        input_id += system
        target += [im_start] + [IGNORE_TOKEN_ID] * (len(system) - 3) + [im_end] + nl_tokens
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            _input_id = tokenizer(role).input_ids + nl_tokens + \
                        tokenizer(sentence["value"]).input_ids + [im_end] + nl_tokens
            input_id += _input_id
            if role == '<|im_start|>user':
                _target = [im_start] + [IGNORE_TOKEN_ID] * (len(_input_id) - 3) + [im_end] + nl_tokens
            elif role == '<|im_start|>assistant':
                _target = [im_start] + [IGNORE_TOKEN_ID] * len(tokenizer(role).input_ids) + \
                          _input_id[len(tokenizer(role).input_ids) + 1:-2] + [im_end] + nl_tokens
            else:
                raise NotImplementedError
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])
    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


if __name__ == "__main__":
    max_len = 30
    CKPT_PATH = "Qwen-1_8B-Chat"
    tokenizer = AutoTokenizer.from_pretrained(CKPT_PATH, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eod_id

    raw_data = get_data()
    sources = [example["conversations"] for example in raw_data]
    ret = preprocess(sources, tokenizer, max_len)
    input_ids = ret["input_ids"][0].tolist()
    attention_mask = ret["attention_mask"][0].tolist()
    labels = ret["labels"][0].tolist()

    # 展示input_ids
    print(f"{input_ids=}\n")
    input_prompt = tokenizer.decode(input_ids)
    print(f"{input_prompt=}\n")
    input_prompt_list = [tokenizer.decode(input_id) for input_id in input_ids]
    print(f"{input_prompt_list=}\n")
    print(input_prompt)

    # 展示attention_mask
    print(f"{attention_mask=}\n")

    # 展示labels
    print(f"{labels=}\n")
    label_text_list = [label if label == IGNORE_TOKEN_ID else tokenizer.decode(label) for label in labels]
    print(f"{label_text_list=}\n")
    for input_text, label_text in zip(input_prompt_list, label_text_list):
        print(f"{repr(input_text)}\t\t{repr(label_text)}")

    # old ==========================
    for k, v in ret.items():
        v = v[0].tolist()
        print(f'{k}={v} len={len(v)}\n')
    print(repr(tokenizer.decode(ret["input_ids"][0].tolist())))

    for input_id, mask, label in zip(ret["input_ids"][0].tolist(), ret["attention_mask"][0].tolist(),
                                     ret["labels"][0].tolist()):
        print(f"{repr(input_id)}\t{repr(mask)}\t{repr(label)}\t{repr(tokenizer.decode(input_id))}\t"
              f"{repr(label if label == IGNORE_TOKEN_ID else tokenizer.decode(label))}")
