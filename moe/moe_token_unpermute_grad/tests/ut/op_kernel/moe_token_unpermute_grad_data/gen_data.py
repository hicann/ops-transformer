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


def get_cpu_data(
    token_num,
    topk,
    hiddensize,
    permuted_tokens,
    sorted_indices1,
    unpermuted_output_d,
    probs,
    flag,
):
    ori_type = permuted_tokens.dtype
    prob_type = probs.dtype
    if flag:
        unpermuted_tokens = torch.zeros(
            token_num * topk, hiddensize, dtype=torch.float32
        )
        unpermuted_tokens = permuted_tokens.index_select(0, sorted_indices1)
        unpermuted_tokens = unpermuted_tokens.reshape(
            -1, topk, permuted_tokens.size(-1)
        )
        probs_grad = torch.sum(
            unpermuted_output_d.unsqueeze(1) * unpermuted_tokens, -1, keepdim=False
        )

        permuted_tokens_grad = (
            unpermuted_output_d.unsqueeze(1) * probs.unsqueeze(-1)
        ).reshape(-1, hiddensize)
        permuted_tokens_grad1 = torch.zeros(
            token_num * topk, hiddensize, dtype=torch.float32
        )
        permuted_tokens_grad1.index_add_(0, sorted_indices1, permuted_tokens_grad)

        return permuted_tokens_grad1.detach().numpy().astype(
            ori_type
        ), probs_grad.detach().numpy().astype(prob_type)
    else:
        permuted_tokens_grad1 = torch.zeros(
            token_num * topk, hiddensize, dtype=torch.float32
        )
        permuted_tokens_grad1.index_add_(0, sorted_indices1, permuted_tokens_grad)
        return permuted_tokens_grad1.detach().numpy().astype(ori_type), None


def gen_data_and_golden(
    token_num, topk, hiddensize, tokendtype, probdtype, flag
):  # flag=True传probs, False不传probs
    permuted_tokens = np.random.uniform(
        -2, 2, [int(token_num * topk), int(hiddensize)]
    ).astype(tokendtype)
    indices = np.random.randint(0, token_num * topk, size=(token_num * topk,)).astype(
        "int32"
    )
    sorted_indices1 = torch.argsort(indices, stable=True).to(torch.int64)  # 传给标杆
    sorted_indices = (
        torch.argsort(sorted_indices1, stable=True).to(torch.int64).numpy()
    )  # 传给算子
    unpermuted_output_d = np.random.uniform(
        -2, 2, [int(token_num), int(hiddensize)]
    ).astype(tokendtype)
    probs = np.random.uniform(-2, 2, [int(token_num), int(topk)]).astype(probdtype)
    permuted_tokens_grad, probs_grad = get_cpu_data(
        token_num,
        topk,
        hiddensize,
        permuted_tokens,
        sorted_indices1,
        unpermuted_output_d,
        probs,
        flag,
    )

    permuted_tokens.tofile("./input_permuted_tokens.bin")
    sorted_indices.tofile("./input_sorted_indices.bin")
    unpermuted_output_d.tofile("./input_unpermuted_output_d.bin")
    if flag:
        probs.tofile("./input_probs.bin")

    permuted_tokens_grad.tofile("./output_permuted_tokens_grad.bin")
    if flag:
        probs_grad.tofile("./output_probs_grad.bin")


if __name__ == "__main__":
    # 清理bin文件
    os.system("rm -rf *.bin")
    gen_data_and_golden(
        int(sys.argv[1]),
        int(sys.argv[2]),
        int(sys.argv[3]),
        sys.argv[4],
        sys.argv[5],
        sys.argv[6],
    )
