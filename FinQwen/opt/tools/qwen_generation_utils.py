# -*- coding: utf-8 -*-
# @file qwen_generation_utils.py
# @author zhangshilong
# @date 2024/7/7

"""
get_stop_words_ids、make_context、decode_tokens都是直接从Qwen代码中拷贝过来的，不要修改
模仿chat接口的逻辑写批量推理接口batch，如果使用chat时GPU利用率不足100%，换成batch能提升速度（注意调整batch_size不要爆显存）
使用时可以通过 from types import MethodType; model.batch = MethodType(batch, model) 添加为实例方法
"""

from typing import List
from typing import Tuple
from typing import Union

import torch
from transformers import PreTrainedTokenizer

# Types.
TokensType = List[int]


def get_stop_words_ids(chat_format, tokenizer):
    if chat_format == "raw":
        stop_words_ids = [tokenizer.encode("Human:"), [tokenizer.eod_id]]
    elif chat_format == "chatml":
        stop_words_ids = [[tokenizer.im_end_id], [tokenizer.im_start_id]]
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")
    return stop_words_ids


def make_context(
        tokenizer: PreTrainedTokenizer,
        query: str,
        history: List[Tuple[str, str]] = None,
        system: str = "",
        max_window_size: int = 6144,
        chat_format: str = "chatml",
):
    if history is None:
        history = []

    if chat_format == "chatml":
        im_start, im_end = "<|im_start|>", "<|im_end|>"
        im_start_tokens = [tokenizer.im_start_id]
        im_end_tokens = [tokenizer.im_end_id]
        nl_tokens = tokenizer.encode("\n")

        def _tokenize_str(role, content):
            return f"{role}\n{content}", tokenizer.encode(
                role, allowed_special=set()
            ) + nl_tokens + tokenizer.encode(content, allowed_special=set())

        system_text, system_tokens_part = _tokenize_str("system", system)
        system_tokens = im_start_tokens + system_tokens_part + im_end_tokens

        raw_text = ""
        context_tokens = []

        for turn_query, turn_response in reversed(history):
            query_text, query_tokens_part = _tokenize_str("user", turn_query)
            query_tokens = im_start_tokens + query_tokens_part + im_end_tokens
            response_text, response_tokens_part = _tokenize_str(
                "assistant", turn_response
            )
            response_tokens = im_start_tokens + response_tokens_part + im_end_tokens

            next_context_tokens = nl_tokens + query_tokens + nl_tokens + response_tokens
            prev_chat = (
                f"\n{im_start}{query_text}{im_end}\n{im_start}{response_text}{im_end}"
            )

            current_context_size = (
                    len(system_tokens) + len(next_context_tokens) + len(context_tokens)
            )
            if current_context_size < max_window_size:
                context_tokens = next_context_tokens + context_tokens
                raw_text = prev_chat + raw_text
            else:
                break

        context_tokens = system_tokens + context_tokens
        raw_text = f"{im_start}{system_text}{im_end}" + raw_text
        context_tokens += (
                nl_tokens
                + im_start_tokens
                + _tokenize_str("user", query)[1]
                + im_end_tokens
                + nl_tokens
                + im_start_tokens
                + tokenizer.encode("assistant")
                + nl_tokens
        )
        raw_text += f"\n{im_start}user\n{query}{im_end}\n{im_start}assistant\n"

    elif chat_format == "raw":
        raw_text = query
        context_tokens = tokenizer.encode(raw_text)
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")

    return raw_text, context_tokens


def _decode_default(
        tokens: List[int],
        *,
        stop_words: List[str],
        eod_words: List[str],
        tokenizer: PreTrainedTokenizer,
        raw_text_len: int,
        verbose: bool = False,
        return_end_reason: bool = False,
        errors: str = 'replace',
):
    trim_decode_tokens = tokenizer.decode(tokens, errors=errors)[raw_text_len:]
    if verbose:
        print("\nRaw Generate: ", trim_decode_tokens)

    end_reason = f"Gen length {len(tokens)}"
    for stop_word in stop_words:
        trim_decode_tokens = trim_decode_tokens.replace(stop_word, "").strip()
    for eod_word in eod_words:
        if eod_word in trim_decode_tokens:
            end_reason = f"Gen {eod_word!r}"
        trim_decode_tokens = trim_decode_tokens.split(eod_word)[0]
    trim_decode_tokens = trim_decode_tokens.strip()
    if verbose:
        print("\nEnd Reason:", end_reason)
        print("\nGenerate: ", trim_decode_tokens)

    if return_end_reason:
        return trim_decode_tokens, end_reason
    else:
        return trim_decode_tokens


def _decode_chatml(
        tokens: List[int],
        *,
        stop_words: List[str],
        eod_token_ids: List[int],
        tokenizer: PreTrainedTokenizer,
        raw_text_len: int,
        context_length: int,
        verbose: bool = False,
        return_end_reason: bool = False,
        errors: str = 'replace'
):
    end_reason = f"Gen length {len(tokens)}"
    eod_token_idx = context_length
    for eod_token_idx in range(context_length, len(tokens)):
        if tokens[eod_token_idx] in eod_token_ids:
            end_reason = f"Gen {tokenizer.decode([tokens[eod_token_idx]])!r}"
            break

    trim_decode_tokens = tokenizer.decode(tokens[:eod_token_idx], errors=errors)[raw_text_len:]
    if verbose:
        print("\nRaw Generate w/o EOD:", tokenizer.decode(tokens, errors=errors)[raw_text_len:])
        print("\nRaw Generate:", trim_decode_tokens)
        print("\nEnd Reason:", end_reason)
    for stop_word in stop_words:
        trim_decode_tokens = trim_decode_tokens.replace(stop_word, "").strip()
    trim_decode_tokens = trim_decode_tokens.strip()
    if verbose:
        print("\nGenerate:", trim_decode_tokens)

    if return_end_reason:
        return trim_decode_tokens, end_reason
    else:
        return trim_decode_tokens


def decode_tokens(
        tokens: Union[torch.LongTensor, TokensType],
        tokenizer: PreTrainedTokenizer,
        raw_text_len: int,
        context_length: int,
        chat_format: str,
        verbose: bool = False,
        return_end_reason: bool = False,
        errors: str = "replace",
) -> str:
    if torch.is_tensor(tokens):
        tokens = tokens.cpu().numpy().tolist()

    if chat_format == "chatml":
        return _decode_chatml(
            tokens,
            stop_words=[],
            eod_token_ids=[tokenizer.im_start_id, tokenizer.im_end_id],
            tokenizer=tokenizer,
            raw_text_len=raw_text_len,
            context_length=context_length,
            verbose=verbose,
            return_end_reason=return_end_reason,
            errors=errors,
        )
    elif chat_format == "raw":
        return _decode_default(
            tokens,
            stop_words=["<|endoftext|>"],
            eod_words=["<|endoftext|>"],
            tokenizer=tokenizer,
            raw_text_len=raw_text_len,
            verbose=verbose,
            return_end_reason=return_end_reason,
            errors=errors,
        )
    else:
        raise NotImplementedError(f"Unknown chat format {chat_format!r}")


def batch(
        model,
        tokenizer: PreTrainedTokenizer,
        queries: List[str],
        system: str = "You are a helpful assistant."
) -> List[str]:
    assert tokenizer.padding_side == "left"

    # 虽然make_context会同时返回context_tokens，但是难以padding，还是得依赖tokenizer
    batch_raw_text = [
        make_context(
            tokenizer, query, system=system,
            max_window_size=model.generation_config.max_window_size,
            chat_format=model.generation_config.chat_format)[0]
        for query in queries
    ]

    batch_input_ids = tokenizer(batch_raw_text, padding=True, return_tensors="pt")["input_ids"].to(model.device)
    stop_words_ids = get_stop_words_ids(model.generation_config.chat_format, tokenizer)

    batch_output_ids = model.generate(
        batch_input_ids,
        stop_words_ids=stop_words_ids,
        return_dict_in_generate=False,
        generation_config=model.generation_config
    )

    input_len = batch_input_ids.shape[1]
    padding_lens = [input_ids.eq(tokenizer.pad_token_id).sum().item() for input_ids in batch_input_ids]

    batch_response = [
        decode_tokens(
            output_ids[padding_len:],
            tokenizer,
            raw_text_len=len(raw_text),
            context_length=input_len - padding_len,
            chat_format=model.generation_config.chat_format,
            verbose=False,
            errors="replace"
        )
        for output_ids, padding_len, raw_text in zip(batch_output_ids, padding_lens, batch_raw_text)
    ]

    return batch_response
