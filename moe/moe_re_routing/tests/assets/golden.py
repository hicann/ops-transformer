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

import numpy

__spec__ = {
    "moe_re_routing": "MoeReRoutingTestSpec",
}


class MoeReRoutingTestSpec:
    @staticmethod
    def golden(
        tokens,
        expert_token_num_per_rank,
        per_token_scales=None,
        *,
        expert_token_num_type=0,
        idx_type=0,
        **kwargs,
    ):
        tokens_dtype = tokens.dtype
        token_num, token_length = tokens.shape
        rank_num, expert_num = expert_token_num_per_rank.shape

        permute_tokens = numpy.zeros((token_num, token_length), dtype=tokens_dtype)
        permute_token_idx = numpy.zeros(token_num, dtype=numpy.int32)

        if per_token_scales is not None:
            if per_token_scales.ndim == 1:
                permute_per_token_scales = numpy.zeros(
                    token_num, dtype=per_token_scales.dtype
                )
            else:
                scalar_length = per_token_scales.shape[1]
                permute_per_token_scales = numpy.zeros(
                    (token_num, scalar_length), dtype=per_token_scales.dtype
                )
        else:
            permute_per_token_scales = None

        expert_token_num = numpy.sum(expert_token_num_per_rank, axis=0)
        per_expert_offset = numpy.zeros(
            expert_num, dtype=expert_token_num_per_rank.dtype
        )
        per_expert_offset[1:] = numpy.cumsum(expert_token_num)[:-1]

        src_offset = 0
        for cur_rank in range(rank_num):
            for expert in range(expert_num):
                num_tokens = expert_token_num_per_rank[cur_rank, expert]
                if num_tokens == 0:
                    continue
                dst_start = per_expert_offset[expert]
                dst_end = dst_start + num_tokens
                src_end = src_offset + num_tokens

                permute_tokens[dst_start:dst_end] = tokens[src_offset:src_end]
                if idx_type == 0:
                    permute_token_idx[dst_start:dst_end] = numpy.arange(
                        src_offset, src_end, dtype=numpy.int32
                    )
                else:
                    permute_token_idx[src_offset:src_end] = numpy.arange(
                        dst_start, dst_end, dtype=numpy.int32
                    )

                if per_token_scales is not None:
                    permute_per_token_scales[dst_start:dst_end] = per_token_scales[
                        src_offset:src_end
                    ]

                src_offset = src_end
                per_expert_offset[expert] = dst_end

        expert_token_num_out = expert_token_num.astype(expert_token_num_per_rank.dtype)
        return [
            permute_tokens,
            permute_per_token_scales,
            permute_token_idx,
            expert_token_num_out,
        ]

    @staticmethod
    def customize_inputs(
        tokens, expert_token_num_per_rank, per_token_scales=None, **kwargs
    ):
        tokens_num = tokens.shape[0]
        rank_num, expert_num = expert_token_num_per_rank.shape
        total_slots = rank_num * expert_num

        if total_slots == 0:
            return (tokens, expert_token_num_per_rank, per_token_scales)

        avg_tokens = tokens_num / total_slots
        min_tokens = max(1, int(avg_tokens * 0.5))
        max_tokens = max(min_tokens + 1, int(avg_tokens * 2))

        new_etr = numpy.zeros(
            (rank_num, expert_num), dtype=expert_token_num_per_rank.dtype
        )
        remaining_tokens, remaining_slots = tokens_num, total_slots

        slot_indices = [(r, e) for r in range(rank_num) for e in range(expert_num)]
        numpy.random.shuffle(slot_indices)

        for rank_idx, expert_idx in slot_indices:
            if remaining_tokens <= 0:
                break
            if remaining_slots == 1:
                alloc = remaining_tokens
            else:
                current_min = max(
                    min_tokens, remaining_tokens - (remaining_slots - 1) * max_tokens, 1
                )
                current_max = min(
                    max_tokens, remaining_tokens - (remaining_slots - 1) * min_tokens
                )
                current_max = max(current_max, current_min + 1)
                alloc = (
                    min_tokens
                    if (current_min >= current_max or remaining_tokens < min_tokens)
                    else numpy.random.randint(current_min, current_max)
                )
            new_etr[rank_idx, expert_idx] = alloc
            remaining_tokens -= alloc
            remaining_slots -= 1

        diff = tokens_num - new_etr.sum()
        if diff != 0:
            non_zero = numpy.argwhere(new_etr > 0)
            if len(non_zero) > 0:
                for _ in range(abs(diff)):
                    idx = numpy.random.choice(len(non_zero))
                    r, e = non_zero[idx]
                    new_etr[r, e] += 1 if diff > 0 else -1

        return (tokens, new_etr, per_token_scales)

    tolerance = {
        "float32": {"standard": "stat_rel_err"},
        "float16": {"standard": "stat_rel_err"},
        "bfloat16": {"standard": "stat_rel_err"},
        "int8": {"standard": "stat_rel_err"},
        "int32": {"standard": "stat_rel_err"},
        "int64": {"standard": "stat_rel_err"},
    }
