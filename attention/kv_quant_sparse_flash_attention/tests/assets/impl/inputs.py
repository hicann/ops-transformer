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

__input__ = {
    "e2e": {"torch_npu.npu_kv_quant_sparse_flash_attention": "generate_kv_quant_inputs"}
}

import math
import numpy as np
import struct
import torch


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


def write_float32_scale_bytes(packed_tensor, d_nope, rope_head_dim, scale_value=1.0):
    """Populate packed per-tile scale bytes in the raw KV cache tail."""
    if packed_tensor is None or not torch.is_tensor(packed_tensor):
        return
    rope_start = d_nope
    scale_start = d_nope + rope_head_dim * 2
    raw_bytes = packed_tensor.view(torch.uint8)
    if raw_bytes.shape[-1] > rope_start:
        raw_bytes[..., rope_start : min(scale_start, raw_bytes.shape[-1])].zero_()
    if raw_bytes.shape[-1] <= scale_start:
        return
    scale_tail = raw_bytes[..., scale_start:]
    byte_values = list(struct.pack("<f", float(scale_value))) * (
        scale_tail.shape[-1] // 4
    )
    if len(byte_values) < scale_tail.shape[-1]:
        byte_values.extend([0] * (scale_tail.shape[-1] - len(byte_values)))
    scale_tail.copy_(
        torch.tensor(
            byte_values[: scale_tail.shape[-1]],
            dtype=torch.uint8,
            device=packed_tensor.device,
        )
    )


def fill_optional_scale(scale_tensor):
    if scale_tensor is not None and torch.is_tensor(scale_tensor):
        scale_tensor.fill_(1.0)


def copy_value_from_key(value, key):
    if (
        value is None
        or key is None
        or not torch.is_tensor(value)
        or not torch.is_tensor(key)
    ):
        return
    if tuple(value.shape) != tuple(key.shape):
        return
    value.copy_(key.to(dtype=value.dtype, device=value.device))


def copy_sparse_indices(dst, src):
    if torch.is_tensor(dst):
        dst.copy_(src.to(dtype=dst.dtype, device=dst.device))
    elif hasattr(dst, "dtype"):
        dst[...] = src.cpu().numpy().astype(dst.dtype, copy=False)
    else:
        dst[:] = src.cpu().tolist()


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


def generate_kv_quant_inputs(
    query,
    key,
    value,
    sparse_indices,
    scale_value,
    key_quant_mode,
    value_quant_mode,
    sparse_block_size,
    **kwargs,
):
    """Generate valid sparse indices and preprocess inputs for KvQuantSparseFlashAttention with KV quantization."""
    if sparse_indices is None:
        return

    layout_query = kwargs.get("layout_query", "BSND")
    layout_kv = kwargs.get("layout_kv", "BSND")
    sparse_mode = kwargs.get("sparse_mode", 0)
    actual_seq_q = kwargs.get("actual_seq_lengths_query")
    actual_seq_kv = kwargs.get("actual_seq_lengths_kv")
    block_table = kwargs.get("block_table")
    tile_size = kwargs.get("tile_size", 128)
    rope_head_dim = kwargs.get("rope_head_dim", 64)
    d_nope = kwargs.get("d_nope", 512)

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

    # Pytest and op docs both exercise MLA-absorb with value cloned from key.
    copy_value_from_key(value, key)
    write_float32_scale_bytes(key, d_nope, rope_head_dim, 1.0)
    write_float32_scale_bytes(value, d_nope, rope_head_dim, 1.0)
    fill_optional_scale(kwargs.get("key_dequant_scale"))
    fill_optional_scale(kwargs.get("value_dequant_scale"))

    act_q_list = to_list(actual_seq_q)
    act_kv_list = to_list(actual_seq_kv)

    if layout_query == "BSND":
        B = query.shape[0]
        S1 = query.shape[1]
    else:
        B = len(act_q_list) if act_q_list else 1
        S1 = max(act_q_list) if act_q_list else query.shape[0]

    if layout_kv in ["BSND", "PA_BSND"]:
        N2 = key.shape[-2] if layout_kv == "BSND" else key.shape[2]
    else:
        N2 = key.shape[1]

    K = sparse_indices.shape[-1]
    sparse_block_size_val = sparse_block_size or 1

    if not act_kv_list:
        act_kv_list = [key.shape[1] if layout_kv == "BSND" else 1024] * B
    if not act_q_list:
        act_q_list = [S1] * B

    sparse_indices_new = torch.full(sparse_indices.shape, -1, dtype=torch.int32)

    if layout_query == "TND" and len(sparse_indices.shape) == 3:
        batch_q_lengths = []
        prev = 0
        for cum_sum in act_q_list:
            batch_q_lengths.append(cum_sum - prev)
            prev = cum_sum

        batch_kv_lengths = (
            act_kv_list
            if layout_kv == "PA_BSND"
            else (
                [
                    cum_sum - prev_kv
                    for prev_kv, cum_sum in zip([0] + act_kv_list[:-1], act_kv_list)
                ]
                if layout_kv == "TND"
                else act_kv_list
            )
        )

        t_offset = 0
        for b in range(B):
            batch_q = (
                batch_q_lengths[b] if b < len(batch_q_lengths) else batch_q_lengths[0]
            )
            batch_kv = (
                batch_kv_lengths[b]
                if b < len(batch_kv_lengths)
                else batch_kv_lengths[0]
            )

            for t_local in range(batch_q):
                t_global = t_offset + t_local
                for n in range(N2):
                    threshold = (
                        batch_kv
                        if sparse_mode == 0
                        else batch_kv - batch_q + t_local + 1
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
                        if topk > 1:
                            sparse_indices_new[t_global, n, : topk - 1] = block_indices[
                                : topk - 1
                            ]
                        sparse_indices_new[t_global, n, topk - 1] = valid_blocks - 1

            t_offset += batch_q
    else:
        for b in range(B):
            kv_len = act_kv_list[b] if b < len(act_kv_list) else act_kv_list[0]
            q_len = act_q_list[b] if b < len(act_q_list) else act_q_list[0]

            for n in range(N2):
                for s in range(S1):
                    threshold = kv_len if sparse_mode == 0 else kv_len - q_len + s + 1
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
                        if topk > 1:
                            sparse_indices_new[b, s, n, : topk - 1] = block_indices[
                                : topk - 1
                            ]
                        sparse_indices_new[b, s, n, topk - 1] = valid_blocks - 1
    copy_sparse_indices(sparse_indices, sparse_indices_new)
