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
import torch_npu
import math
import numpy as np
import logging
import pytest
from test_case import TestCases
from test_utils import (
    generate_qkv,
    generate_pse,
    generate_npu_mask,
    trans_bnsd_to_layout,
    get_seqlen_list,
)
from cpu_impl import tforward
from npu_impl import fa_npu

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)


def check_result(expect, result, test_name):
    logger.info(f"Start comparing precision of {test_name}.")
    ratio_threshold = 0.005
    threshold_diff = 0.005
    if expect.shape == result.shape:
        if torch.all(torch.eq(expect, result)):
            logger.info(f"{test_name} results are fully consistent.")
            ratio_diff = 0
            diff = 0
            return 1, 0, 0
        else:
            diff = torch.abs(expect.sub(result))
            min = torch.empty(expect.shape, dtype=torch.float16)
            min.fill_(0.000025)
            max = torch.max(torch.abs(expect), torch.abs(result))
            threshold = torch.max(max.mul(ratio_threshold), min)
            mask = diff > threshold
            num_diff = torch.sum(mask)
            ratio_diff = num_diff / torch.numel(expect)
            if ratio_diff > threshold_diff:
                logger.warning(
                    f"warning: {test_name} results have {num_diff} elements exceeding threshold "
                    f"{ratio_threshold:.2%}, ratio is {ratio_diff:.2%}!"
                )
            else:
                logger.info(
                    f"info: {test_name} results have {num_diff} elements exceeding threshold "
                    f"{ratio_threshold:.2%}, ratio is {ratio_diff:.2%}."
                )
            ratio = (1 - ratio_diff).cpu().detach().numpy()
            max = torch.max(diff).cpu().detach().numpy()
            logger.info(f"{test_name} diff_max: {max:.8f}")
            sum = torch.sum(diff).cpu().detach().numpy()
            logger.info(f"{test_name} diff_sum: {sum:.8f}")
    else:
        logger.error(
            f"error: {test_name} results are wrong, shape does not match expected!"
        )
        logger.error(f"error: expect shape: {expect.shape}")
        logger.error(f"error: result shape: {result.shape}")
        ratio = 0
        max = 999999
        sum = 999999
    return ratio, max, sum


def call_flash_attn(test_name, **kwargs):
    b = kwargs.get("B", 1)  # layout=TND, b=1
    n1 = kwargs.get("N1")
    n2 = kwargs.get("N2", n1)
    sq = kwargs.get("Sq", -1)
    skv = kwargs.get("Skv", sq)
    d = kwargs.get("D")
    d_v = kwargs.get("DV", d)
    d_rope = kwargs.get("DRope", 0)
    input_layout = kwargs.get("input_layout")
    scale = kwargs.get("scale", 1 / (d**0.5))
    pse_type = kwargs.get("pse_type", 1)
    pse_layout = kwargs.get("pse_layout", "none").lower()
    keep_prob = kwargs.get("keep_prob", 1)
    q_start_idx = kwargs.get("q_start_idx", 0)
    kv_start_idx = kwargs.get("kv_start_idx", 0)
    dtype = kwargs.get("dtype", torch.bfloat16)
    sparse_mode = kwargs.get("sparse_mode", None)
    pre_tokens = kwargs.get("pre_tokens", 2147483647)
    next_tokens = kwargs.get("next_tokens", 2147483647)
    prefix = kwargs.get("prefix", [])
    pse_b = b
    pse_s1 = sq
    pse_s2 = skv
    actual_seq_qlen = None
    actual_seq_kvlen = None
    if input_layout == "TND":
        actual_seq_qlen = list(kwargs.get("actual_seq_qlen"))
        actual_seq_kvlen = list(kwargs.get("actual_seq_kvlen", actual_seq_qlen))
        seqlen_q_list = get_seqlen_list(actual_seq_qlen)
        seqlen_k_list = get_seqlen_list(actual_seq_kvlen)
        sq = max(actual_seq_qlen)
        skv = max(actual_seq_kvlen)
        pse_b = len(actual_seq_qlen)
        pse_s1 = seqlen_q_list.max()
        pse_s2 = seqlen_k_list.max()
        if pse_layout in ["bnhs", "1nhs"]:
            pse_s1 = max(1024, pse_s1)

    pse_cpu, pse_npu = generate_pse(
        pse_b,
        n1,
        pse_s1,
        pse_s2,
        pse_type,
        pse_layout,
        dtype,
        q_start_idx,
        kv_start_idx,
    )
    q, k, v, q_rope, k_rope, qf, kf = generate_qkv(
        b, n1, n2, sq, skv, d, d_v, d_rope, input_layout, dtype
    )
    out, x_max, x_sum = tforward(qf, kf, v, pse_cpu, **kwargs)

    atten_mask = generate_npu_mask(
        b, sq, skv, sparse_mode, pre_tokens, next_tokens, prefix
    )
    npu_out, npu_max, npu_sum = fa_npu(
        q, k, v, q_rope, k_rope, atten_mask, pse_npu, **kwargs
    )

    out = trans_bnsd_to_layout(out, input_layout)
    check_result(out.float(), npu_out.float(), "out")
    check_result(x_max.contiguous().view(-1), npu_max.contiguous().view(-1), "max")
    check_result(x_sum.contiguous().view(-1), npu_sum.contiguous().view(-1), "sum")


def test_npu_flash_attn():
    for test_name, test_data in TestCases.items():
        logger.info(f"================Run test case: {test_name} begin===============")
        call_flash_attn(test_name, **test_data)
        logger.info(f"================Run test case: {test_name} end=================")
