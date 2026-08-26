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

"""Input customization for FlashAttn TTK cases.

E2E customize_inputs contract: modify tensors in-place via x.copy_(value).
No return value. Tensor shapes/dtypes are pre-allocated by TTK from CSV.

Rule: only fill tensors that are provided (not None). Do not auto-generate
missing parameters — use them if present, skip if absent.
"""

import torch


def _to_int_list(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return [int(x) for x in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(value)]


def _fill_block_table(block_table, seqused_kv, block_size):
    if block_table is None or not torch.is_tensor(block_table):
        return
    if seqused_kv is None:
        return
    if block_size is None or int(block_size) <= 0:
        return

    shape = tuple(block_table.shape)
    if len(shape) < 2:
        return

    bt = torch.full(shape, -1, dtype=torch.int32)
    seq_list = _to_int_list(seqused_kv)
    if not seq_list:
        return

    bs = int(block_size)
    block_idx = 0
    for b in range(shape[0]):
        kv_len = seq_list[b] if b < len(seq_list) else seq_list[-1]
        num_blocks_b = (kv_len + bs - 1) // bs
        for j in range(min(num_blocks_b, shape[1])):
            bt[b][j] = block_idx
            block_idx += 1
    block_table.copy_(bt.to(dtype=block_table.dtype, device=block_table.device))


def _fill_attn_mask(attn_mask, mask_mode, win_left=-1, win_right=-1):
    if attn_mask is None or not torch.is_tensor(attn_mask):
        return
    if mask_mode == 0:
        return

    # mask_mode=3/4: attn_mask 固定为 2048x2048 上三角矩阵, 对角线全 0
    mask = torch.triu(torch.ones(2048, 2048), diagonal=1)
    attn_mask.copy_(mask.to(dtype=attn_mask.dtype, device=attn_mask.device))


def zero_metadata(metadata):
    if metadata is None:
        return
    if torch.is_tensor(metadata):
        metadata.zero_()
    else:
        metadata[...] = 0


def customize_inputs(
    q,
    k,
    v,
    *,
    block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    batch_size=None,
    sinks=None,
    attn_mask=None,
    metadata=None,
    softmax_scale=1.0,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    max_seqlen_q=None,
    max_seqlen_kv=None,
    layout_q="BSND",
    layout_kv="BSND",
    layout_out="BSND",
    return_softmax_lse=False,
    **kwargs,
):
    """Fill auxiliary tensors in-place for FlashAttn TTK cases.

    All data is generated on CPU then copied to the target tensor's device.
    Only fills tensors that are provided; missing parameters are left as-is.
    """
    block_size = kwargs.get("block_size")

    # 小整数张量(cu_seqlens/seqused)的值经 *_values attr 传入(避免与
    # TTK match_overload 的输入计数冲突), 必须先于 block_table 填充,
    # 否则 _fill_block_table 会读到随机值导致分页表错误
    for name, tensor in (
        ("cu_seqlens_q", cu_seqlens_q),
        ("cu_seqlens_kv", cu_seqlens_kv),
        ("seqused_q", seqused_q),
        ("seqused_kv", seqused_kv),
    ):
        values = kwargs.get(f"{name}_values")
        if tensor is not None and values is not None:
            t = torch.tensor(list(values), dtype=tensor.dtype)
            tensor.copy_(t.to(device=tensor.device))

    if torch.is_tensor(block_table):
        _fill_block_table(block_table, seqused_kv, block_size)

    if mask_mode in (3, 4):
        _fill_attn_mask(attn_mask, mask_mode, win_left, win_right)

    zero_metadata(metadata)
