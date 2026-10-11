#!/usr/bin/python
# -*- coding: utf-8 -*-
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
import numpy as np
import torch


def permute_with_ep_grad(
    permuted_tokens: torch.Tensor,
    sorted_indices1: torch.Tensor,
    top_k: int = 1,
    probs: torch.Tensor = None,
    start: int = 0,
    end: int = 0,
):
    num_out_tokens = sorted_indices1.size(0) // top_k
    sorted_indices = torch.sort(sorted_indices1, stable=True)[1]
    res_indices = sorted_indices[start:end]
    tokens_out = torch.zeros(num_out_tokens, permuted_tokens.size(1))
    tokens_out.index_add_(0, res_indices // top_k, permuted_tokens)
    probs_out = torch.zeros(num_out_tokens * top_k)
    if probs is not None:
        probs_out.index_add_(0, res_indices, probs)
        probs_out = probs_out.reshape(num_out_tokens, top_k)
    return tokens_out, probs_out


def get_cpu_data(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    top_k,
    probs: torch.Tensor = None,
):
    permuted_tokens = permuted_tokens.float()
    if probs is not None:
        probs = probs.float()
    unpermuted_tokens = permute_with_ep_grad(
        permuted_tokens, sorted_indices, top_k, probs, 0, sorted_indices.size(0)
    )
    return unpermuted_tokens.bfloat16()


def gen_data_and_golden(
    tokens_num, topk, hidden_size, haveProbs
):  # flag=True传probs, False不传probs
    permuted_tokens = torch.rand(tokens_num * topk, hidden_size)
    sorted_indices_golden = torch.randint(
        low=0, high=tokens_num * topk, size=(tokens_num * topk)
    )  # 传给标杆
    if haveProbs:
        probs = torch.rand(tokens_num, topk)
    else:
        probs = None
    unpermuted_output = get_cpu_data(
        permuted_tokens, sorted_indices_golden, topk, probs
    )

    permuted_tokens = permuted_tokens.numpy()
    sorted_indices = (
        torch.argsort(sorted_indices_golden, stable=True).to(torch.int64).numpy()
    )  # 传给算子
    unpermuted_output = unpermuted_output.numpy()

    permuted_tokens.tofile("./input_permuted_tokens.bin")
    sorted_indices.tofile("./input_sorted_indices.bin")
    if haveProbs:
        probs = probs.numpy()
        probs.tofile("./input_probs.bin")
    unpermuted_output.tofile("./output_unpermuted_tokens.bin")


if __name__ == "__main__":
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(
        int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    )
