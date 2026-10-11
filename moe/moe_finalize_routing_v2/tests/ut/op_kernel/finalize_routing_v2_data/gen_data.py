#!/usr/bin/python3
# -*- coding:utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import sys
import os
import numpy as np
import copy
import re

# import torch
import tensorflow as tf

bf16 = tf.bfloat16.as_numpy_dtype
# torch.manual_seed(0)
np.random.seed(5)


def moe_finalize_routing_v2(
    expanded_permuted_rows: np.ndarray,
    skip1: np.ndarray,
    skip2_optional: np.ndarray,
    bias: np.ndarray,
    scales: np.ndarray,
    expanded_src_to_dst_row: np.ndarray,
    expert_for_source_row: np.ndarray,
) -> np.ndarray:
    # if skip1.dtype == bf16:
    #     expanded_permuted_rows.astype(np.float32)
    #     skip1.astype(np.float32)
    #     bias.astype(np.float32)
    #     scales.astype(np.float32)
    num_rows = expert_for_source_row.shape[0]
    H = expanded_permuted_rows.shape[1]
    if (skip1 is not None) and (skip2_optional is not None):
        out = skip1 + skip2_optional
    elif (skip2_optional is not None) and (skip1 is None):
        out = copy.deepcopy(skip2_optional)
    elif (skip2_optional is None) and (skip1 is not None):
        out = copy.deepcopy(skip1)
    else:
        out = np.zeros([num_rows, H])

    K = expanded_src_to_dst_row.shape[0] // num_rows

    for i in range(num_rows):
        for k in range(K):
            index = expanded_src_to_dst_row[k * num_rows + i]
            if index == -1:
                dst_row = 0
            else:
                dst_row = expanded_permuted_rows[
                    expanded_src_to_dst_row[k * num_rows + i], :
                ]
            expert_id = expert_for_source_row[i, k]
            if bias is not None:
                out[i, :] += scales[i, k] * (dst_row + bias[expert_id, :])
            else:
                out[i, :] += scales[i, k] * dst_row

    return out


# def moe_finalize_routing(expanded_permuted_rows,
#                          skip1,
#                          skip2_optional,
#                          bias,
#                          scales,
#                          expanded_src_to_dst_row,
#                          expert_for_source_row):
#     out = skip1
#     if skip2_optional is not None:
#         out = skip1 + skip2_optional
#     num_rows = skip1.shape[0]
#     K = expanded_src_to_dst_row.shape[0] // num_rows

#     expanded_src_to_dst_row = expanded_src_to_dst_row.view(K, num_rows).T
#     for i in range(num_rows):
#         dst_idx = expanded_src_to_dst_row[i].long()
#         dst_row = expanded_permuted_rows[dst_idx, :]
#         expert_id = expert_for_source_row[i, :].long()
#         out[i, :] =  out[i, :] + (scales[i, :].view(K, -1) * (dst_row + bias[expert_id, :])).sum(dim=0)

#     return out


def gen_data_and_golden(
    expert_num, token_len, top_k, num_rows, d_type="float32", use_skip2=True
):
    """
    @param: expert_num->E
    @param: token_len:H
    @param: top_k: K
    @param: num_rows: B*S
    """
    d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": bf16,
        "bf16fp32": bf16,
    }
    scale_d_type_dict = {
        "float32": np.float32,
        "float16": np.float16,
        "bfloat16": bf16,
        "bf16fp32": np.float32,
    }
    dtype = d_type_dict.get(d_type)
    scale_dtype = scale_d_type_dict.get(d_type)
    if use_skip2.lower() == "true":
        use_skip2 = True
    else:
        use_skip2 = False
    print("#" * 100)
    print("generate moe finalize routing golden input params:")
    print("E: ", expert_num)
    print("H: ", token_len)
    print("K: ", top_k)
    print("BS: ", num_rows)
    print("#" * 100)
    expanded_permuted_rows = np.random.randn(num_rows * top_k, token_len).astype(dtype)
    skip1 = np.random.randn(num_rows, token_len).astype(dtype)
    if use_skip2:
        skip2_optional = np.random.randn(num_rows, token_len).astype(dtype)
    else:
        skip2_optional = None
    bias = np.random.randn(expert_num, token_len).astype(dtype)
    scales = np.random.randn(num_rows, top_k).astype(scale_dtype)
    expanded_src_to_dst_row = np.arange(num_rows * top_k).astype(np.int32)
    np.random.shuffle(expanded_src_to_dst_row)
    expert_for_source_row = np.random.randint(
        low=0, high=expert_num, size=(num_rows, top_k)
    ).astype(np.int32)
    expanded_permuted_rows.tofile("expanded_permuted_rows.bin")
    skip1.tofile("skip1.bin")
    skip2_optional.tofile("skip2_optional.bin")
    bias.tofile("bias.bin")
    scales.tofile("scales.bin")
    expanded_src_to_dst_row.tofile("expanded_src_to_dst_row.bin")
    expert_for_source_row.tofile("expert_for_source_row.bin")

    # if dtype == np.float32 or dtype == np.float16 :
    out_golden = moe_finalize_routing_v2(
        expanded_permuted_rows,
        skip1,
        skip2_optional,
        bias,
        scales,
        expanded_src_to_dst_row,
        expert_for_source_row,
    )
    out_golden = out_golden.tofile("out_golden.bin")


# def gen_data_and_golden(expert_num, token_len, top_k, num_rows, d_type="float32", use_skip2=True):
#     """
#     @param: expert_num->E
#     @param: token_len:H
#     @param: top_k: K
#     @param: num_rows: B*S
#     """
#     d_type_dict = {
#         "float32": torch.float32,
#         "float16": torch.float16,
#         "bfloat16": torch.bfloat16
#     }
#     dtype = d_type_dict.get(d_type)
#     if use_skip2.lower() == 'true':
#         use_skip2 = True
#     else:
#         use_skip2 = False
#     print("#"*100)
#     print("generate moe finalize routing golden input params:")
#     print("E: ", expert_num)
#     print("H: ", token_len)
#     print("K: ", top_k)
#     print("BS: ", num_rows)
#     print("#"*100)
#     expanded_permuted_rows = torch.randn(num_rows * top_k, token_len, dtype=dtype)
#     skip1 = torch.randn(num_rows, token_len, dtype=dtype)
#     if use_skip2:
#         skip2_optional = torch.randn(num_rows, token_len, dtype=dtype)
#     else:
#         skip2_optional = None
#     bias = torch.randn(expert_num, token_len, dtype=dtype)
#     scales = torch.randn(num_rows, top_k, dtype=dtype)
#     expanded_src_to_dst_row = torch.randint(low=0, high=num_rows*top_k, size=(num_rows*top_k),dtype=torch.int32)
#     expert_for_source_row = torch.randint(low=0, high=expert_num, size=(num_rows, top_k),dtype=torch.int32)
#     torch.save(expanded_permuted_rows,f"expanded_permuted_rows.bin")
#     torch.save(skip1,f"skip1.bin")
#     torch.save(skip2_optional,f"skip2_optional.bin")
#     torch.save(bias,f"bias.bin")
#     torch.save(scales,f"scales.bin")
#     torch.save(expanded_src_to_dst_row,f"expanded_src_to_dst_row.bin")
#     torch.save(expert_for_source_row,f"expert_for_source_row.bin")

#     out_golden = moe_finalize_routing(expanded_permuted_rows, skip1, skip2_optional, bias, scales, expanded_src_to_dst_row, expert_for_source_row)
#     out_golden = torch.save(out_golden,f"out_golden.bin")


if __name__ == "__main__":
    # 清理bin文件
    os.system("rm -rf *.bin")
    E, H, K, BS, d_type, use_skip2 = (
        sys.argv[1],
        sys.argv[2],
        sys.argv[3],
        sys.argv[4],
        sys.argv[5],
        sys.argv[6],
    )
    E, H, K, BS = int(E), int(H), int(K), int(BS)
    gen_data_and_golden(E, H, K, BS, d_type=d_type, use_skip2=use_skip2)
