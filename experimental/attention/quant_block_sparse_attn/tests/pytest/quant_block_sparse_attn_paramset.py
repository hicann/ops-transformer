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

import torch


COMMON = {
    "Testcase_Name": [None],
    "layout_kv": ["PA_BNSD"],
    "output_dtype": [torch.bfloat16],
    "N1": [32],
    "N2": [4],
    "D": [128],
    "sparse_q_block_size": [128],
    "sparse_kv_block_size": [128],
    "softmax_scale": [0.088388],
    "layout_sparse_indices": ["B_N_Qb_Kb"],
    "layout_out": ["TND"],
    "quant_mode": [1],
    "mask_mode": [3],
    "pa_block_padding_bytes": [0],
    "S1EQS2": [False],
}


def _case(**kwargs):
    params = dict(COMMON)
    params.update(kwargs)
    return params


TEST_PARAMS = {
    "TND_TYPICAL_CASE": _case(
        layout_q=["TND"],
        B=[4],
        S1=[1024],
        S2=[1024],
        sparse_count=[4],
        return_softmax_lse=[True],
        actlen_mode=["full"],
        sparse_pattern=["sequential"],
        block_table_pattern=["sequential"],
        p_scale_value=[1.0],
        seed=[3],
    ),
    "NTD_TYPICAL_CASE": _case(
        layout_q=["NTD"],
        B=[2],
        S1=[555],
        S2=[666],
        sparse_count=[4],
        return_softmax_lse=[True],
        actlen_mode=["full"],
        sparse_pattern=["sequential"],
        block_table_pattern=["sequential"],
        p_scale_value=[1.0],
        seed=[3],
    ),
    "tnd_dense_equiv": _case(
        layout_q=["TND"],
        B=[1],
        S1=[128],
        S2=[256],
        sparse_count=[2],
        return_softmax_lse=[True],
        actlen_mode=["full"],
        sparse_pattern=["sequential"],
        block_table_pattern=["sequential"],
        p_scale_value=[1.0],
        seed=[3],
    ),
    "tnd_random_actlen": _case(
        layout_q=["TND"],
        B=[2],
        S1=[128],
        S2=[256],
        sparse_count=[2],
        return_softmax_lse=[True],
        actlen_mode=["random"],
        sparse_pattern=["random"],
        block_table_pattern=["random"],
        p_scale_value=[1.0],
        seed=[17],
    ),
}

ENABLED_PARAMS = [TEST_PARAMS["TND_TYPICAL_CASE"], TEST_PARAMS["NTD_TYPICAL_CASE"]]
# ENABLED_PARAMS = [TEST_PARAMS[key] for key in TEST_PARAMS.keys()]
