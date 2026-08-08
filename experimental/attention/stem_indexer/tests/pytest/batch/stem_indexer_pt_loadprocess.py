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
import torch_npu

import stem_indexer_golden
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import custom_ops  # noqa: E402, F401
from stem_indexer_aclgraph import build_metadata, call_stem_indexer_graph  # noqa: E402


def torch_load_cpu(filepath):
    try:
        return torch.load(filepath, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(filepath, map_location="cpu")


def move_inputs_to_npu(test_data):
    return {
        "qflat": test_data["qflat"].npu(),
        "kflat": test_data["kflat"].npu(),
        "vbias": test_data["vbias"].npu(),
        "q_seq_lens": test_data["q_seq_lens"].npu(),
        "kv_seq_lens": test_data["kv_seq_lens"].npu(),
        "num_prompt_tokens": test_data["num_prompt_tokens"].npu(),
    }


def call_stem_indexer(case, npu_inputs):
    """eager 模式：直接调用 torch.ops.custom.npu_stem_indexer。"""
    metadata = build_metadata(case, npu_inputs)
    return torch.ops.custom.npu_stem_indexer(
        npu_inputs["qflat"],
        npu_inputs["kflat"],
        npu_inputs["vbias"],
        npu_inputs["q_seq_lens"],
        npu_inputs["kv_seq_lens"],
        num_prompt_tokens=npu_inputs["num_prompt_tokens"],
        metadata=metadata,
        **stem_indexer_golden.get_call_attrs(case),
    )


def stem_indexer_process(filepath, device_id=0, return_test_data=False, mode="eager"):
    test_data = torch_load_cpu(filepath)
    case = test_data["case"]
    print(f"Running StemIndexer testcase (mode={mode}):", filepath)
    torch_npu.npu.set_device(device_id)
    npu_inputs = move_inputs_to_npu(test_data)

    if mode == "graph":
        npu_result = call_stem_indexer_graph(case, npu_inputs, device_id=device_id)
    else:
        npu_result = call_stem_indexer(case, npu_inputs)

    torch_npu.npu.synchronize()
    result = (
        test_data["expected_sparse_indices"],
        test_data["expected_sparse_seq_len"],
        npu_result,
        case,
    )
    if return_test_data:
        return (*result, test_data)
    return result
