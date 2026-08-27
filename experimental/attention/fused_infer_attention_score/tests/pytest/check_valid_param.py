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

import math
import random
import logging
import torch

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)


def validate_config(params):
    (
        batch_size,
        q_head_num,
        kv_head_num,
        q_seq,
        kv_seq,
        head_dim,
        dtype,
        in_layout,
        rope,
    ) = params
    # 校验
    if dtype not in [torch.bfloat16, torch.float16]:
        raise ValueError("dtype should be: float16/bfloat16")
    if in_layout not in ["BSND", "BNSD"]:
        raise ValueError("input_layout should be: BSND/BNSD")


def check_result(expect, result):
    attn_out_diff = result.reshape(-1) - expect.reshape(-1)
    thres = 0.005
    idx = torch.nonzero(abs(attn_out_diff) > thres).squeeze()
    values = attn_out_diff[abs(attn_out_diff) > thres]

    if idx.numel() > 0:
        logger.info(f"element indices where diff > {thres}: {idx}")
        logger.info(f"element values where diff > {thres}: {values}")

    pass_num = torch.sum(abs(attn_out_diff.squeeze()) <= thres)
    total_num = attn_out_diff.numel()
    accuracy = pass_num / total_num
    logger.info(f"pass num: {pass_num}, total num: {total_num}, accuracy: {accuracy}")

    assert torch.allclose(result.flatten(), expect.flatten(), rtol=0.05, atol=0.05)
