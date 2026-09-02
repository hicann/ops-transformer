#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import torch
from einops import rearrange


def generate_cpu_mask(
    b, s1, s2, mask_mode, win_left, win_right, prefix=None, index=None, band_index=None
):
    if mask_mode is None or mask_mode == 0:
        return None
    elif mask_mode == 3:
        mask = torch.triu(torch.ones(s1, s2), diagonal=s2 - s1 + 1).bool()
    elif mask_mode == 4:
        # -1 means infinity (no mask on that side)
        effective_win_right = s2 if win_right == -1 else win_right
        effective_win_left = s1 if win_left == -1 else win_left
        atten_mask_u = torch.triu(
            torch.ones(s1, s2), diagonal=effective_win_right + 1 + s2 - s1
        )
        atten_mask_l = torch.tril(
            torch.ones(s1, s2), diagonal=-effective_win_left - 1 + s2 - s1
        )
        mask = (atten_mask_u + atten_mask_l).bool()
    return mask


def page_attn_for_bnsd(bnsd_tensor, b, act_seq_lens_kv, block_table, block_size):
    block_num = int(block_table.max()) + 1
    kv_cache_bnsd_shape = (
        block_num,
        bnsd_tensor.shape[1],
        block_size,
        bnsd_tensor.shape[3],
    )
    page_cache_tensor = torch.zeros(size=kv_cache_bnsd_shape, dtype=bnsd_tensor.dtype)
    actual_b = block_table.shape[0]
    is_tnd = b == 1 and actual_b > 1
    cum_kv_offset = 0
    for i in range(actual_b):
        b_seq = act_seq_lens_kv[i] if len(act_seq_lens_kv) > i else act_seq_lens_kv[0]
        b_block_num = (b_seq + block_size - 1) // block_size
        if is_tnd:
            batch_dim = 0
            seq_offset = cum_kv_offset
        else:
            batch_dim = i
            seq_offset = 0
        for j in range(b_block_num):
            start_idx = seq_offset + j * block_size
            end_idx = seq_offset + min((j + 1) * block_size, b_seq)
            actual_size = end_idx - start_idx
            slice_data = bnsd_tensor[batch_dim, :, start_idx:end_idx, :]
            page_cache_tensor[block_table[i][j], :, :actual_size, :] = slice_data
        cum_kv_offset += b_seq
    return page_cache_tensor


def dtype_sizeof(data_type):
    if data_type == "fp16" or data_type == "bf16":
        return 2
    elif data_type == "int8" or data_type == "fp8":
        return 1


def rearrange_by_block_table(
    bnsd_tensor, block_table, block_size, b, act_seq_lens_kv, kv_storage_mode, kv_dtype
):
    page_cache_tensor = page_attn_for_bnsd(
        bnsd_tensor, b, act_seq_lens_kv, block_table, block_size
    )
    if kv_storage_mode == "PA_BBND":
        return page_cache_tensor.permute(0, 2, 1, 3)
    elif kv_storage_mode == "PA_BNBD":
        return page_cache_tensor
    elif (
        kv_storage_mode == "PA_NZ"
    ):  # PA_NZ: shape=(NumBlocks, N_kv, BlockSize/16, D, 16)，分页注意力NZ格式
        blk_elem = 32 // dtype_sizeof(kv_dtype)
        page_cache_tensor = page_cache_tensor.reshape(
            page_cache_tensor.shape[0],
            page_cache_tensor.shape[1],
            page_cache_tensor.shape[2],
            page_cache_tensor.shape[3] // blk_elem,
            blk_elem,
        ).permute(0, 1, 3, 2, 4)
        return page_cache_tensor
    else:
        return None


def trans_bnsd_to_layout(tensor, layout_type, **kwargs):
    b = kwargs.get("B")
    block_size = kwargs.get("block_size")
    block_table = kwargs.get("block_table")
    seqused_kv = kwargs.get("seqused_kv")
    dtype = kwargs.get("Dtype", "bf16")

    if tensor is None:
        return None
    else:
        if layout_type == "BSH":
            return rearrange(tensor.clone(), "b n s d -> b s (n d)")
        elif layout_type == "SBH":
            return rearrange(tensor.clone(), "b n s d -> s b (n d)")
        elif layout_type == "BSND":
            return rearrange(tensor.clone(), "b n s d -> b s n d")
        elif layout_type == "TND":
            return rearrange(tensor.clone(), "1 n s d -> s n d")
        elif (
            layout_type == "PA_BBND"
            or layout_type == "PA_BNBD"
            or layout_type == "PA_NZ"
        ):
            return rearrange_by_block_table(
                tensor, block_table, block_size, b, seqused_kv, layout_type, dtype
            )
        return tensor.clone()


def make_noncontiguous(tensor, nc_dims):
    """返回指定维非连续但底层数据相同的 tensor view.

    Args:
        tensor: 连续的 torch.Tensor
        nc_dims: 0     --> 仅 dim0 非连续 (pad dim1 后切片)
                 (0,1) --> dim0+dim1 都非连续 (pad dim2 后切片)

    Note: dim1-only 物理上不可行——dim1 stride 膨胀必然导致 dim0 stride 联动;
          测试用例中所有 dim1 场景已移除。

    Returns:
        非连续 tensor view (shape 不变, .is_contiguous() == False)
    """
    device = tensor.device
    dtype = tensor.dtype
    shape = tensor.shape

    if nc_dims == 0:
        pad_dim = 1
        shp = list(shape)
        shp[pad_dim] += 1
        p = torch.zeros(shp, dtype=dtype, device=device)
        ws = [slice(None)] * len(shape)
        ws[pad_dim] = slice(None, shape[pad_dim])
        p[tuple(ws)] = tensor
        return p[tuple(ws)]
    elif nc_dims == (0, 1):
        pad_dim = 2
        shp = list(shape)
        shp[pad_dim] += 1
        p = torch.zeros(shp, dtype=dtype, device=device)
        ws = [slice(None)] * len(shape)
        ws[pad_dim] = slice(None, shape[pad_dim])
        p[tuple(ws)] = tensor
        return p[tuple(ws)]
    return tensor
