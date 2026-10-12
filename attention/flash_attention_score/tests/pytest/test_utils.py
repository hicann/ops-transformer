#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

import torch
import math
import numpy as np
import random
from einops import rearrange

PHILOX_M4_32 = [0xD2511F53, 0xCD9E8D57]
PHIL0X_W_32 = [0x9E3779B9, 0xBB67AE85]


def philox4_32(counter, key, rounds):
    return philox(
        counter,
        key,
        philox4_round,
        PHILOX_M4_32,
        philox4_bumpkey,
        PHIL0X_W_32,
        32,
        0xFFFFFFFF,
        rounds,
    )


def philox4_round(counter, key, philox_m, len_w, mask_w):
    prod = philox_m[0] * counter[0]
    hi_1 = prod >> len_w
    lo_1 = prod & mask_w
    prod = philox_m[1] * counter[2]
    hi_2 = prod >> len_w
    lo_2 = prod & mask_w
    counter[0] = hi_2 ^ counter[1] ^ key[0]
    counter[1] = lo_2
    counter[2] = hi_1 ^ counter[3] ^ key[1]
    counter[3] = lo_1


def philox4_bumpkey(key, philox_w, mask_w):
    key[0] = (key[0] + philox_w[0]) & mask_w
    key[1] = (key[1] + philox_w[1]) & mask_w


def philox(
    counter,
    key,
    philox_round,
    philox_m,
    philox_bumpkey,
    philox_w,
    len_w,
    mask_w,
    rounds,
):
    for i in range(rounds - 1):
        philox_round(counter, key, philox_m, len_w, mask_w)
        philox_bumpkey(key, philox_w, mask_w)
    philox_round(counter, key, philox_m, len_w, mask_w)
    return counter


def inc_counter(counter):
    for i in range(4):
        counter[i] = (counter[i] + 1) & 0xFFFFFFFF
        if counter[i] != 0:
            return counter
    return counter


def philox_random(rounds, counter, key, count):
    ret = list()
    for i in range((count + 255) // 256 * 256 // 4):
        ret.extend(philox4_32(counter[:], key[:], rounds))
        counter = inc_counter(counter[:])
    return np.array(ret)[:count]


def uint32_to_uint8_array(data):
    little_endian = np.array([1], dtype=np.uint32).view(np.uint8)[0] == 1
    uint8_array = np.zeros((4), dtype=np.uint8)
    if little_endian:
        uint8_array[0] = data & 0xFF
        uint8_array[1] = (data >> 8) & 0xFF
        uint8_array[2] = (data >> 16) & 0xFF
        uint8_array[3] = (data >> 24) & 0xFF
    else:
        uint32_data = np.array(data, dtype=np.dtype(">u4"))
        uint8_array[0] = uint32_data.view(np.uint8)[3]
        uint8_array[1] = uint32_data.view(np.uint8)[2]
        uint8_array[2] = uint32_data.view(np.uint8)[1]
        uint8_array[3] = uint32_data.view(np.uint8)[0]
    return uint8_array


def gen_dropmask_uint8(rounds, counter, key, uint8_count, keep_prob):
    keep_prob_uint8 = keep_prob * 255
    uint32_count = math.ceil(uint8_count / 4)
    philox_uint32 = philox_random(rounds, counter, key, uint32_count)
    philox_uint8 = []
    golden_uint8 = []
    for each_uint32 in philox_uint32:
        four_uint8 = uint32_to_uint8_array(each_uint32)
        philox_uint8.extend(four_uint8)
    for each_philox_uint8 in philox_uint8:
        compare_res = 1 if each_philox_uint8 <= keep_prob_uint8 else 0
        golden_uint8.append(compare_res)
    return golden_uint8[:uint8_count]


def gen_dropmask(b, n1, s1, s2, keep_prob=0.8, seed=2, offset=0):
    rounds = 7
    seed_high = seed // 0x100000000
    seed_low = seed - seed_high * 0x100000000
    key = [seed_low, seed_high]
    offset_aligned = math.ceil(offset / 16)
    offset_high = offset_aligned // 0x100000000
    offset_low = offset_aligned - offset_high * 0x100000000
    counter = [offset_low, offset_high, 0, 0]
    s2_aligned = math.ceil(s2 / 16) * 16
    count = b * n1 * s1 * s2_aligned
    golden_uint8 = gen_dropmask_uint8(rounds, counter, key, count, keep_prob)
    drop_mask_np = (
        np.array(golden_uint8)
        .reshape(b, n1, s1, s2_aligned)[:, :, :, : s2 - s2_aligned]
        .astype(np.uint8)
        if s2_aligned != s2
        else np.array(golden_uint8).reshape(b, n1, s1, s2).astype(np.uint8)
    )
    drop_mask = torch.from_numpy(drop_mask_np)
    return drop_mask


def gen_dropmask_tnd(length, keep_prob=0.8, seed=2, offset=0):
    rounds = 7
    seed_high = seed // 0x100000000
    seed_low = seed - seed_high * 0x100000000
    key = [seed_low, seed_high]
    offset_aligned = math.ceil(offset / 16)
    offset_high = offset_aligned // 0x100000000
    offset_low = offset_aligned - offset_high * 0x100000000
    counter = [offset_low, offset_high, 0, 0]
    count = math.ceil(length / 16) * 16
    golden_uint8 = gen_dropmask_uint8(rounds, counter, key, count, keep_prob)
    drop_mask_np = np.array(golden_uint8).astype(np.uint8)
    drop_mask = torch.from_numpy(drop_mask_np)
    return drop_mask


def trans_bnsd_to_layout(tensor, input_layout):
    if tensor is None:
        return None
    else:
        if input_layout == "BSH":
            return rearrange(tensor.clone(), "b n s d -> b s (n d)")
        elif input_layout == "SBH":
            return rearrange(tensor.clone(), "b n s d -> s b (n d)")
        elif input_layout == "BSND":
            return rearrange(tensor.clone(), "b n s d -> b s n d")
        elif input_layout == "TND":
            return rearrange(tensor.clone(), "1 n s d -> s n d")
        else:
            return tensor.clone()


def get_seqlen_list(actual_seqlen):
    seq_list = torch.zeros(len(actual_seqlen), dtype=torch.int64)
    seq_list[0] = actual_seqlen[0]
    for i in range(1, len(actual_seqlen)):
        seq_list[i] = actual_seqlen[i] - actual_seqlen[i - 1]
    return seq_list


def generate_pse(b, n1, s1, s2, pse_type, pse_layout, dtype, q_start_idx, kv_start_idx):
    if pse_layout.lower() == "none" or pse_layout is None:
        return None, None
    pse_layout = pse_layout.lower()
    if pse_type in [0, 1]:
        assert pse_layout in ["bnss", "bn1s", "1nss", "bnhs", "1nhs"]
        if pse_layout == "bnss":
            pse = torch.randn(b, n1, s1, s2).to(dtype)
            pse_npu = pse
        elif pse_layout == "bn1s":
            pse = torch.randn(b, n1, 1, s2).to(dtype)
            pse_npu = pse
        elif pse_layout == "1nss":
            pse = torch.randn(1, n1, s1, s2).to(dtype)
            pse_npu = pse
        elif pse_layout == "bnhs":
            pse_bias = (
                1 / (torch.arange(b) + 1).unsqueeze(1) * 1 / (torch.arange(n1) + 1)
            )
            bn11 = pse_bias.reshape(b, n1, 1, 1)
            alibi_base = -torch.abs(
                torch.arange(s1).unsqueeze(1)
                - torch.arange(s2).unsqueeze(0)
                - (s1 - s2)
            )
            pse = bn11 * alibi_base
            pse_npu = pse[:, :, -1024:, :]
        elif pse_layout == "1nhs":
            pse_bias = 1 / (torch.arange(n1) + 1)
            bn11 = pse_bias.reshape(1, n1, 1, 1)
            alibi_base = -torch.abs(
                torch.arange(s1).unsqueeze(1)
                - torch.arange(s2).unsqueeze(0)
                - (s1 - s2)
            )
            pse = bn11 * alibi_base
            pse_npu = pse[:, :, -1024:, :]
        return pse, pse_npu
    elif pse_type in [2, 3]:
        # alibi
        assert pse_layout in ["bn", "n"]
        slopes_size = n1 if pse_layout == "n" else b * n1
        if pse_layout == "n":
            alibi_slopes = torch.randn(n1).float()
            alibi_bias = alibi_slopes.reshape(1, n1, 1, 1)
        else:
            alibi_slopes = torch.randn(b, n1).float()
            alibi_bias = alibi_slopes.reshape(b, n1, 1, 1)
        if pse_type == 2:
            alibi_base = -torch.abs(
                torch.arange(s2).unsqueeze(0)
                - torch.arange(s1).unsqueeze(1)
                - q_start_idx
                + kv_start_idx
            )
        else:
            alibi_base = -(
                torch.abs(
                    torch.arange(s2).unsqueeze(0)
                    - torch.arange(s1).unsqueeze(1)
                    - q_start_idx
                    + kv_start_idx
                ).sqrt()
            )
        return alibi_bias * alibi_base, alibi_slopes


def generate_npu_mask(b, s1, s2, sparse_mode, pre_tokens, next_tokens, prefix=None):
    if sparse_mode is None:
        return None
    if sparse_mode == 0:
        atten_mask_u = torch.triu(torch.ones(s1, s2), diagonal=next_tokens + 1)
        atten_mask_l = torch.tril(torch.ones(s1, s2), diagonal=-pre_tokens - 1)
        mask = (atten_mask_u + atten_mask_l).bool()
    elif sparse_mode == 1:
        mask = torch.zeros(s1, s2).bool()
    elif sparse_mode in [2, 3, 4, 7, 8]:
        mask = torch.triu(torch.ones(2048, 2048), diagonal=1).bool()
    elif sparse_mode == 5:
        assert prefix is not None
        mask = torch.triu(torch.ones(b, 1, s1, s2), diagonal=s2 - s1 + 1)
        for i in range(0, b):
            mask[i, :, :, : prefix[i]] = 0
        mask = mask.bool()
    elif sparse_mode == 6:
        upper = torch.triu(torch.ones(2048, 2048), diagonal=1)
        lower = torch.cat((torch.zeros(1024, 1024), torch.ones(1024, 1024)), dim=1)
        mask = torch.cat((upper, lower), dim=0).bool()
    return mask


def generate_cpu_mask(
    b,
    s1,
    s2,
    sparse_mode,
    pre_tokens,
    next_tokens,
    prefix=None,
    index=None,
    band_index=None,
):
    if sparse_mode is None:
        return None
    if sparse_mode == 0:
        atten_mask_u = torch.triu(torch.ones(s1, s2), diagonal=next_tokens + 1)
        atten_mask_l = torch.tril(torch.ones(s1, s2), diagonal=-pre_tokens - 1)
        mask = (atten_mask_u + atten_mask_l).bool()
    elif sparse_mode == 1:
        mask = torch.zeros(s1, s2).bool()
    elif sparse_mode == 2:
        mask = torch.triu(torch.ones(s1, s2), diagonal=1).bool()
    elif sparse_mode == 3:
        mask = torch.triu(torch.ones(s1, s2), diagonal=s2 - s1 + 1).bool()
    elif sparse_mode == 4:
        atten_mask_u = torch.triu(
            torch.ones(s1, s2), diagonal=next_tokens + 1 + s2 - s1
        )
        atten_mask_l = torch.tril(
            torch.ones(s1, s2), diagonal=-pre_tokens - 1 + s2 - s1
        )
        mask = (atten_mask_u + atten_mask_l).bool()
    elif sparse_mode == 5 or sparse_mode == 6:
        assert prefix is not None
        mask = torch.triu(torch.ones(b, 1, s1, s2), diagonal=s2 - s1 + 1)
        for i in range(0, b):
            mask[i, :, :, : prefix[i]] = 0
        mask = mask.bool()
    elif sparse_mode == 6:
        upper = torch.triu(torch.ones(2048, 2048), diagonal=1)
        lower = torch.cat((torch.zeros(1024, 1024), torch.ones(1024, 1024)), dim=1)
        mask = torch.cat((upper, lower), dim=0).bool()
    elif sparse_mode in [7, 8]:
        if index == band_index:
            atten_mask_u = torch.triu(
                torch.ones(s1, s2), diagonal=next_tokens + 1 + s2 - s1
            )
            atten_mask_l = torch.tril(
                torch.ones(s1, s2), diagonal=-pre_tokens - 1 + s2 - s1
            )
            mask = (atten_mask_u + atten_mask_l).bool()
        else:
            mask = torch.triu(
                torch.ones(s1, s2), diagonal=s2 - s1 + 1 if sparse_mode == 7 else 1
            ).bool()
    return mask


def generate_qkv(b, n1, n2, s1, s2, d, d_v, d_rope, input_layout, dtype):
    q = torch.randn(
        b, n1, s1, d, generator=torch.Generator().manual_seed(42), dtype=dtype
    )
    k = torch.randn(
        b, n2, s2, d, generator=torch.Generator().manual_seed(43), dtype=dtype
    )
    v = torch.randn(
        b, n2, s2, d_v, generator=torch.Generator().manual_seed(44), dtype=dtype
    )

    # MLA:d=128, d_rope=64
    q_rope = torch.randn(
        b, n1, s1, d_rope, generator=torch.Generator().manual_seed(45), dtype=dtype
    )
    k_rope = torch.randn(
        b, n2, s2, d_rope, generator=torch.Generator().manual_seed(46), dtype=dtype
    )

    qf = torch.cat((q, q_rope), -1)
    kf = torch.cat((k, k_rope), -1)

    return q, k, v, q_rope, k_rope, qf, kf
