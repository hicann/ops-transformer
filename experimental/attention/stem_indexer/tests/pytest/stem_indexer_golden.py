#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import math

import torch


HEAD_DIM = 128
MAX_TOPK_COUNT = 256
K_SMALL_SEQ_MAX = 56
K_MEDIUM_SEQ_MAX = 160
K_BLOCK_NUM_RATE_MEDIUM = 0.2
K_BLOCK_NUM_BIAS_MEDIUM = 30
K_BLOCK_NUM_RATE_LARGE = 0.1
K_BLOCK_NUM_BIAS_LARGE = 30
METADATA_HEADER_SIZE = 16
METADATA_CORE_STRIDE = 16
METADATA_AIC_CORE_NUM = 36
METADATA_AIV_CORE_NUM = 72
METADATA_ALIGN_SIZE = 4096
DEFAULT_TOPK_SCORE_PRECISION = 1
TOPK_SCORE_TYPE_BY_PRECISION = {1: "uint32", 2: "uint16"}

DTYPE_MAP = {
    "BF16": torch.bfloat16,
    "FP32": torch.float32,
    "FP16": torch.float16,
    "INT32": torch.int32,
}


def ceil_div(value, factor):
    return (int(value) + int(factor) - 1) // int(factor)


def get_metadata_size(batch_size, kv_heads):
    max_section_num = int(batch_size) * int(kv_heads)
    metadata_size = METADATA_HEADER_SIZE + max_section_num * (
        METADATA_AIC_CORE_NUM + METADATA_AIV_CORE_NUM
    ) * METADATA_CORE_STRIDE
    return ceil_div(metadata_size, METADATA_ALIGN_SIZE) * METADATA_ALIGN_SIZE


def to_float32(value):
    return float(torch.tensor(value, dtype=torch.float32).item())


def fma_float32(multiplier, multiplicand, addend):
    # Emulate one correctly rounded float32 multiply-add used by the AICore scalar expression.
    return to_float32(float(multiplier) * float(multiplicand) + float(addend))


def parse_special_setting(setting):
    parsed = {}
    if not setting:
        return parsed
    for item in str(setting).split(";"):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def parse_shape(shape_str):
    return [
        int(item.strip()) for item in shape_str.strip()[1:-1].split(",") if item.strip()
    ]


def dtype_from_name(dtype_name):
    if dtype_name not in DTYPE_MAP:
        raise ValueError(
            f"Unsupported dtype name in StemIndexer test case: {dtype_name}"
        )
    return DTYPE_MAP[dtype_name]


def get_topk_score_precision(case):
    precision = int(case.get("topk_score_precision", DEFAULT_TOPK_SCORE_PRECISION))
    if precision not in TOPK_SCORE_TYPE_BY_PRECISION:
        raise ValueError(
            f"topk_score_precision must be 1(uint32) or 2(uint16), but got {precision}"
        )
    return precision


def get_golden_topk_score_type(case):
    return TOPK_SCORE_TYPE_BY_PRECISION[get_topk_score_precision(case)]


def round_float32_to_bfloat16(scores):
    return scores.to(torch.bfloat16).to(torch.float32)


def get_topk_sort_scores(scores, score_type):
    if score_type == "uint16":
        return round_float32_to_bfloat16(scores)
    return scores


def make_float_tensor(shape, dtype, pattern, seed):
    if pattern == "zeros":
        return torch.zeros(shape, dtype=dtype)
    if pattern == "constant_positive":
        return torch.full(shape, 0.125, dtype=dtype)
    if pattern == "strictly_ascending_by_k_block":
        base = torch.arange(shape[-1], dtype=torch.float32).reshape(
            *([1] * (len(shape) - 1)), shape[-1]
        )
        return base.expand(shape).contiguous().to(dtype)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    data = torch.rand(shape, generator=generator, dtype=torch.float32) * 2.0 - 1.0
    return data.to(dtype)


def get_tensor_shapes(case):
    special = parse_special_setting(case.get("special_setting", ""))
    batch_size = case["batch_size"]
    q_heads = case["q_heads"]
    kv_heads = case["kv_heads"]
    stem_block_size = case["stem_block_size"]
    stem_stride = case["stem_stride"]
    flatten_dim = stem_stride * HEAD_DIM
    max_qb = ceil_div(max(case["q_seq_lens"]), stem_block_size)
    max_kb = ceil_div(max(case["kv_seq_lens"]), stem_block_size)

    qflat_shape = (
        parse_shape(special["qflat_shape"])
        if "qflat_shape" in special
        else [batch_size, q_heads, max_qb, flatten_dim]
    )
    kflat_shape = (
        parse_shape(special["kflat_shape"])
        if "kflat_shape" in special
        else [batch_size, kv_heads, max_kb, flatten_dim]
    )
    vbias_shape = [batch_size, kv_heads, max_kb]
    metadata_shape = (
        parse_shape(special["metadata_shape"])
        if "metadata_shape" in special
        else [get_metadata_size(batch_size, kv_heads)]
    )
    return qflat_shape, kflat_shape, vbias_shape, metadata_shape


def build_case_inputs(case):
    special = parse_special_setting(case.get("special_setting", ""))
    qflat_shape, kflat_shape, vbias_shape, metadata_shape = get_tensor_shapes(case)

    qflat_pattern = special.get("qflat", "random_uniform_seeded")
    kflat_pattern = special.get("kflat", "random_uniform_seeded")
    vbias_pattern = special.get("vbias", "random_uniform_seeded")

    qflat = make_float_tensor(
        qflat_shape, dtype_from_name(case["qflat_dtype"]), qflat_pattern, seed=17
    )
    kflat = make_float_tensor(
        kflat_shape, dtype_from_name(case["kflat_dtype"]), kflat_pattern, seed=23
    )
    vbias = make_float_tensor(
        vbias_shape, dtype_from_name(case["vbias_dtype"]), vbias_pattern, seed=31
    )
    q_seq_lens = torch.tensor(case["q_seq_lens"], dtype=torch.int32)
    kv_seq_lens = torch.tensor(case["kv_seq_lens"], dtype=torch.int32)
    num_prompt_tokens = torch.tensor(case["num_prompt_tokens"], dtype=torch.int32)
    metadata = torch.zeros(metadata_shape, dtype=torch.int32)

    return {
        "qflat": qflat,
        "kflat": kflat,
        "vbias": vbias,
        "q_seq_lens": q_seq_lens,
        "kv_seq_lens": kv_seq_lens,
        "num_prompt_tokens": num_prompt_tokens,
        "metadata": metadata,
    }


def get_call_attrs(case):
    return {
        "causal": bool(case["causal"]),
        "stem_block_size": int(case["stem_block_size"]),
        "stem_stride": int(case["stem_stride"]),
        "alpha": float(case["alpha"]),
        "initial_blocks": int(case["initial_blocks"]),
        "window_size": int(case["window_size"]),
        "k_block_num_rate_medium": K_BLOCK_NUM_RATE_MEDIUM,
        "k_block_num_bias_medium": K_BLOCK_NUM_BIAS_MEDIUM,
        "k_block_num_rate_large": K_BLOCK_NUM_RATE_LARGE,
        "k_block_num_bias_large": K_BLOCK_NUM_BIAS_LARGE,
        "topk_score_precision": get_topk_score_precision(case),
    }


def get_metadata_attrs(case):
    return {
        "causal": bool(case["causal"]),
        "stem_block_size": int(case["stem_block_size"]),
        "dim_qkflat": int(case["stem_stride"]) * HEAD_DIM,
        "window_size": int(case["window_size"]),
    }


def is_decode_case(q_len, kv_len, prompt_len):
    return q_len == 1 and prompt_len >= kv_len


def calc_topk_budget(case, q_block_idx, q_block_num, kv_block_num, prompt_len):
    prompt_block_num = ceil_div(prompt_len, case["stem_block_size"])
    if prompt_block_num < K_SMALL_SEQ_MAX:
        k_start = prompt_block_num
    elif prompt_block_num < K_MEDIUM_SEQ_MAX:
        k_start = int(
            to_float32(
                to_float32(prompt_block_num) * to_float32(K_BLOCK_NUM_RATE_MEDIUM)
            )
            + to_float32(K_BLOCK_NUM_BIAS_MEDIUM)
        )
    else:
        k_start = int(
            to_float32(
                to_float32(prompt_block_num) * to_float32(K_BLOCK_NUM_RATE_LARGE)
            )
            + to_float32(K_BLOCK_NUM_BIAS_LARGE)
        )

    s1_pos = q_block_idx + kv_block_num - q_block_num
    decay_len = prompt_block_num - k_start
    if s1_pos < k_start or decay_len <= 1:
        return k_start

    k_start_float = to_float32(k_start)
    alpha = to_float32(case["alpha"])
    k_end = to_float32(k_start_float * alpha)
    t = to_float32(to_float32(s1_pos - k_start) / to_float32(decay_len - 1))
    decay_delta = to_float32(k_end - k_start_float)
    interpolated_topk_count = fma_float32(t, decay_delta, k_start_float)
    topk_count = math.floor(interpolated_topk_count)
    return min(max(topk_count, 1), k_start)


def get_forced_indices(s2_valid, initial_blocks, window_size):
    sink = range(0, min(initial_blocks, s2_valid))
    window_begin = max(s2_valid - window_size, 0)
    window = range(window_begin, s2_valid)
    return set(sink).union(window)


def calc_causal_s2_valid(q_block_idx, q_block_num, kv_block_num):
    s2_valid = kv_block_num - q_block_num + q_block_idx + 1
    return max(min(s2_valid, kv_block_num), 0)


def stem_indexer_golden(case, inputs):
    qflat = inputs["qflat"].float()
    kflat = inputs["kflat"].float()
    vbias = inputs["vbias"].float()

    batch_size = case["batch_size"]
    q_heads = case["q_heads"]
    kv_heads = case["kv_heads"]
    g_size = q_heads // kv_heads
    q_seq_lens = case["q_seq_lens"]
    kv_seq_lens = case["kv_seq_lens"]
    prompt_lens = case["num_prompt_tokens"]
    stem_block_size = case["stem_block_size"]
    stem_stride = case["stem_stride"]
    score_scale = 1.0 / ((stem_block_size // stem_stride) ** 2)
    topk_score_type = get_golden_topk_score_type(case)

    max_qb = qflat.shape[2]
    max_kb = kflat.shape[2]
    sparse_indices = torch.full(
        (batch_size, q_heads, max_qb, max_kb), -1, dtype=torch.int32
    )
    sparse_seq_len = torch.zeros((batch_size, q_heads, max_qb), dtype=torch.int32)

    for batch_idx in range(batch_size):
        q_len = q_seq_lens[batch_idx]
        kv_len = kv_seq_lens[batch_idx]
        prompt_len = prompt_lens[batch_idx]
        q_block_num = ceil_div(q_len, stem_block_size)
        kv_block_num = ceil_div(kv_len, stem_block_size)
        decode = is_decode_case(q_len, kv_len, prompt_len)

        for kv_head_idx in range(kv_heads):
            q_head_begin = kv_head_idx * g_size
            q_head_end = q_head_begin + g_size
            q_group = qflat[batch_idx, q_head_begin:q_head_end].reshape(
                g_size * max_qb, -1
            )
            k_group = kflat[batch_idx, kv_head_idx]
            score_group = torch.matmul(q_group, k_group.transpose(0, 1)) * score_scale
            score_group = score_group.reshape(g_size, max_qb, max_kb)
            score_group = score_group + vbias[batch_idx, kv_head_idx].reshape(
                1, 1, max_kb
            )

            for local_q_head_idx, q_head_idx in enumerate(
                range(q_head_begin, q_head_end)
            ):
                for q_block_idx in range(q_block_num):
                    if case["causal"] and not decode:
                        s2_valid = calc_causal_s2_valid(
                            q_block_idx, q_block_num, kv_block_num
                        )
                    else:
                        s2_valid = kv_block_num
                    if s2_valid <= 0:
                        continue

                    topk_select_num = min(
                        max(
                            calc_topk_budget(
                                case, q_block_idx, q_block_num, kv_block_num, prompt_len
                            ),
                            0,
                        ),
                        MAX_TOPK_COUNT,
                    )
                    forced = get_forced_indices(
                        s2_valid, case["initial_blocks"], case["window_size"]
                    )
                    candidate_indices = [
                        idx for idx in range(s2_valid) if idx not in forced
                    ]
                    scores = score_group[local_q_head_idx, q_block_idx]
                    sort_scores = get_topk_sort_scores(scores, topk_score_type)
                    ranked = sorted(
                        candidate_indices,
                        key=lambda idx: (-float(sort_scores[idx]), idx),
                    )
                    selected = sorted(forced.union(ranked[:topk_select_num]))
                    sparse_seq_len[batch_idx, q_head_idx, q_block_idx] = len(selected)
                    if selected:
                        sparse_indices[
                            batch_idx, q_head_idx, q_block_idx, : len(selected)
                        ] = torch.tensor(selected, dtype=torch.int32)

    return sparse_indices, sparse_seq_len
