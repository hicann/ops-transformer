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

"""
Input plugin for sparse_flash_attention - generates valid sparse_indices.
"""

import math
import numpy as np
import torch

__input__ = {
    "e2e": {"torch_npu.npu_sparse_flash_attention": "generate_valid_sparse_indices"}
}


def to_list(value):
    if value is None:
        return []
    if torch.is_tensor(value):
        value = value.detach().cpu().tolist()
    elif hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [int(v) for v in value]
    return [int(value)]


def fill_tensor_from_value(tensor, value):
    if tensor is None or value is None or not torch.is_tensor(tensor):
        return
    data = torch.tensor(value, dtype=tensor.dtype, device=tensor.device)
    tensor.copy_(data.reshape(tensor.shape))


def fill_random_block_table(block_table, actual_seq_kv, block_size, block_num, seed=0):
    if block_table is None:
        return
    shape = tuple(block_table.shape)
    if len(shape) < 2:
        return
    block_size = max(int(block_size or 1), 1)
    actual = to_list(actual_seq_kv)
    if not actual:
        actual = [shape[1] * block_size] * shape[0]
    used_blocks = [
        math.ceil(
            max(int(actual[idx] if idx < len(actual) else actual[-1]), 0) / block_size
        )
        for idx in range(shape[0])
    ]
    required_blocks = sum(used_blocks)
    block_num = int(block_num or required_blocks)
    if required_blocks > block_num:
        raise ValueError(
            f"block_table requires {required_blocks} blocks, but block_num is {block_num}"
        )

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed or 0))
    block_ids = torch.randperm(block_num, dtype=torch.int32, generator=generator)
    if torch.is_tensor(block_table):
        table = torch.full(shape, -1, dtype=torch.int32, device=block_table.device)
    else:
        table = np.full(shape, -1, dtype=np.int32)
    cursor = 0
    for batch_idx, blocks in enumerate(used_blocks):
        take = min(blocks, shape[1])
        values = block_ids[cursor : cursor + take]
        if torch.is_tensor(block_table):
            table[batch_idx, :take] = values.to(device=block_table.device)
        else:
            table[batch_idx, :take] = values.cpu().numpy()
        cursor += blocks
    if torch.is_tensor(block_table):
        block_table.copy_(table.to(dtype=block_table.dtype))
    else:
        block_table[...] = table.astype(block_table.dtype, copy=False)


def align_value_with_key_for_mla(key, value):
    if (
        key is None
        or value is None
        or not torch.is_tensor(key)
        or not torch.is_tensor(value)
    ):
        return
    if tuple(key.shape) != tuple(value.shape):
        return
    # Pytest MLA-absorb path feeds value_cache from key_cache; keep TTK NPU input
    # aligned with the golden formula instead of using an independent random V.
    value.copy_(key.to(dtype=value.dtype, device=value.device))


def generate_valid_sparse_indices(
    query, key, value, sparse_indices, scale_value, sparse_block_size, **kwargs
):
    """Generate valid sparse_indices based on threshold (matches pytest logic)."""
    align_value_with_key_for_mla(key, value)

    if sparse_indices is None:
        return

    layout_query = kwargs.get("layout_query", "BSND")
    layout_kv = kwargs.get("layout_kv", "BSND")
    sparse_mode = kwargs.get("sparse_mode", 0)
    actual_seq_q = kwargs.get("actual_seq_lengths_query")
    actual_seq_kv = kwargs.get("actual_seq_lengths_kv")
    block_table = kwargs.get("block_table")

    fill_tensor_from_value(actual_seq_q, kwargs.get("actual_seq_lengths_query"))
    fill_tensor_from_value(actual_seq_kv, kwargs.get("actual_seq_lengths_kv"))
    if layout_kv == "PA_BSND":
        pa_block_size = kwargs.get("block_size") or key.shape[1]
        pa_block_num = kwargs.get("block_num") or key.shape[0]
        fill_random_block_table(
            block_table,
            actual_seq_kv,
            pa_block_size,
            pa_block_num,
            kwargs.get("block_table_seed", 0),
        )

    actual_seq_q_list = to_list(actual_seq_q)
    actual_seq_kv_list = to_list(actual_seq_kv)

    sparse_block_size_val = sparse_block_size or 1

    if layout_query == "BSND":
        B = query.shape[0]
        S1 = query.shape[1]
    else:
        B = len(actual_seq_q_list) if actual_seq_q_list else 1
        S1 = max(actual_seq_q_list) if actual_seq_q_list else query.shape[0]

    N2 = key.shape[-2] if layout_kv in ["BSND", "PA_BSND"] else key.shape[1]
    K = sparse_indices.shape[-1]

    if not actual_seq_kv_list:
        if layout_kv == "BSND":
            default_kv = key.shape[1]
        elif layout_kv == "PA_BSND":
            pa_block_size = kwargs.get("block_size") or key.shape[1]
            bt_width = (
                block_table.shape[1]
                if torch.is_tensor(block_table) and block_table.dim() > 1
                else 1
            )
            default_kv = pa_block_size * bt_width
        else:
            default_kv = key.shape[0]
        actual_seq_kv_list = [default_kv] * B
    if not actual_seq_q_list:
        actual_seq_q_list = [S1] * B

    sparse_indices_new = torch.full(sparse_indices.shape, -1, dtype=torch.int32)

    if layout_query == "TND" and len(sparse_indices.shape) == 3:
        batch_q_lengths = []
        prev = 0
        for cum_sum in actual_seq_q_list:
            batch_q_lengths.append(cum_sum - prev)
            prev = cum_sum

        batch_kv_lengths = (
            actual_seq_kv_list
            if layout_kv == "PA_BSND"
            else [
                cum_sum - prev_kv
                for prev_kv, cum_sum in zip(
                    [0] + actual_seq_kv_list[:-1], actual_seq_kv_list
                )
            ]
            if layout_kv == "TND"
            else actual_seq_kv_list
        )

        t_offset = 0
        for b in range(B):
            batch_seq_q = (
                batch_q_lengths[b] if b < len(batch_q_lengths) else batch_q_lengths[0]
            )
            batch_seq_kv = (
                batch_kv_lengths[b]
                if b < len(batch_kv_lengths)
                else batch_kv_lengths[0]
            )

            for t_local in range(batch_seq_q):
                t_global = t_offset + t_local
                for n in range(N2):
                    threshold = (
                        batch_seq_kv
                        if sparse_mode == 0
                        else batch_seq_kv - batch_seq_q + t_local + 1
                    )
                    if threshold <= 0:
                        continue

                    valid_blocks = math.ceil(max(0, threshold) / sparse_block_size_val)
                    if valid_blocks <= 0:
                        continue

                    if valid_blocks > 1:
                        block_indices = torch.randperm(valid_blocks, dtype=torch.int32)
                    else:
                        block_indices = torch.zeros(1, dtype=torch.int32)

                    topk = min(valid_blocks, K)
                    if topk > 0:
                        sparse_indices_new[t_global, n, : topk - 1] = (
                            block_indices[: topk - 1] if topk > 1 else torch.tensor([])
                        )
                        sparse_indices_new[t_global, n, topk - 1] = valid_blocks - 1

            t_offset += batch_seq_q
    else:
        for b in range(B):
            seq_kv = (
                actual_seq_kv_list[b]
                if b < len(actual_seq_kv_list)
                else actual_seq_kv_list[0]
            )
            seq_q = (
                actual_seq_q_list[b]
                if b < len(actual_seq_q_list)
                else actual_seq_q_list[0]
            )

            for n in range(N2):
                for s in range(S1):
                    threshold = seq_kv if sparse_mode == 0 else seq_kv - seq_q + s + 1
                    if threshold <= 0:
                        continue

                    valid_blocks = math.ceil(max(0, threshold) / sparse_block_size_val)
                    if valid_blocks <= 0:
                        continue

                    if valid_blocks > 1:
                        block_indices = torch.randperm(valid_blocks, dtype=torch.int32)
                    else:
                        block_indices = torch.zeros(1, dtype=torch.int32)

                    topk = min(valid_blocks, K)
                    if topk > 0:
                        sparse_indices_new[b, s, n, : topk - 1] = (
                            block_indices[: topk - 1] if topk > 1 else torch.tensor([])
                        )
                        sparse_indices_new[b, s, n, topk - 1] = valid_blocks - 1

    sparse_indices[:] = sparse_indices_new.to(sparse_indices.dtype).to(
        sparse_indices.device
    )
