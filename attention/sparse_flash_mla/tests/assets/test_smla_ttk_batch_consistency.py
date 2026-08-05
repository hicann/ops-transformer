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

"""White-box regression tests for the SMLA batch assets."""

import importlib.util
from pathlib import Path

import numpy as np
import torch


def load_asset_module(stem):
    path = Path(__file__).with_name("impl") / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"smla_batch_test_{stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMPARE_MODULE = load_asset_module("compare")
INPUTS_MODULE = load_asset_module("inputs")


def batch_kwargs(slices, seed):
    return {
        "batch_consistency_id": ((tuple(
            f"{seed}_0_{start}_{stop}_{step}" for start, stop, step in slices
        ),),),
        "batch_axis": ((0,),),
        "batch_slice_info": ((tuple(slices),),),
        "batch_seed": ((tuple(seed for _ in slices),),),
    }


def test_same_case_batch_compare_accepts_equal_slices():
    output = np.array([[1.0, 2.0], [1.0, 2.0]], dtype=np.float32)

    result = COMPARE_MODULE.compare(
        output, output.copy(), **batch_kwargs(((0, 1, 1), (1, 2, 1)), 7001)
    )

    assert result[-1] == {"pass": True, "precision": "batch_intra=PASS"}


def test_same_case_batch_compare_rejects_different_slices():
    output = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)

    result = COMPARE_MODULE.compare(
        output, output.copy(), **batch_kwargs(((0, 1, 1), (1, 2, 1)), 7001)
    )

    assert result[-1]["pass"] is False
    assert result[-1]["precision"] == "batch_intra=FAIL"


def test_same_case_batch_compare_rejects_different_storage_bits():
    output = np.array([[0.0], [-0.0]], dtype=np.float32)

    result = COMPARE_MODULE.compare(
        output, output.copy(), **batch_kwargs(((0, 1, 1), (1, 2, 1)), 7001)
    )

    assert result[-1]["pass"] is False
    assert result[-1]["precision"] == "batch_intra=FAIL"


def test_batch_random_context_repeats_declared_batch_slices():
    q = torch.empty((3, 1, 1, 1), dtype=torch.float32)
    fields = batch_kwargs(((0, 1, 1), (2, 3, 1)), 7001)
    fields.update({"layout_q": "BSND", "layout_kv": "BSND"})
    original_rand = torch.rand

    with INPUTS_MODULE.TorchBatchRandomContext.from_case(q, None, None, fields):
        value = torch.rand(3, 2)

    assert torch.rand is original_rand
    assert torch.equal(value[0], value[2])


def test_bsnd_batch_context_ignores_auxiliary_q_prefix():
    q = torch.empty((2, 1, 1, 1), dtype=torch.float32)
    fields = batch_kwargs(((0, 1, 1), (1, 2, 1)), 7001)
    fields.update({
        "layout_q": "BSND",
        "layout_kv": "BSND",
        "cu_seqlens_q_values": [0, 14, 28],
    })

    context = INPUTS_MODULE.TorchBatchRandomContext.from_case(q, None, None, fields)

    assert context.q_prefix is None
    assert context.batch_size == 2


def test_tnd_batch_random_context_maps_q_and_kv_token_ranges():
    q = torch.empty((8, 1, 1), dtype=torch.float32)
    ori_kv = torch.empty((12, 1, 1), dtype=torch.float32)
    cmp_kv = torch.empty((4, 1, 1), dtype=torch.float32)
    fields = batch_kwargs(((0, 2, 1), (4, 6, 1)), 7001)
    fields.update({
        "layout_q": "TND",
        "layout_kv": "TND",
        "cu_seqlens_q_values": [0, 2, 4, 6, 8],
        "cu_seqlens_ori_kv_values": [0, 3, 6, 9, 12],
        "cu_seqlens_cmp_kv_values": [0, 1, 2, 3, 4],
        "seqused_q_values": [2, 2, 2, 2],
        "seqused_ori_kv_values": [3, 3, 3, 3],
        "seqused_cmp_kv_values": [1, 1, 1, 1],
        "cmp_residual_kv_values": [0, 0, 0, 0],
    })

    with INPUTS_MODULE.TorchBatchRandomContext.from_case(
            q, ori_kv, cmp_kv, fields):
        q_value = torch.rand(8, 2)
        ori_value = torch.rand(12, 2)
        cmp_value = torch.rand(4, 2)
        batch_value = torch.rand(4, 2, 2)

    assert torch.equal(q_value[0:2], q_value[4:6])
    assert torch.equal(ori_value[0:3], ori_value[6:9])
    assert torch.equal(cmp_value[0:1], cmp_value[2:3])
    assert torch.equal(batch_value[0], batch_value[2])


def test_tnd_batch_random_context_rejects_partial_logical_batch():
    q = torch.empty((8, 1, 1), dtype=torch.float32)
    fields = batch_kwargs(((1, 3, 1), (4, 6, 1)), 7001)
    fields.update({
        "layout_q": "TND",
        "layout_kv": "PA_BBND",
        "cu_seqlens_q_values": [0, 2, 4, 6, 8],
    })

    try:
        INPUTS_MODULE.TorchBatchRandomContext.from_case(q, None, None, fields)
    except ValueError as error:
        assert "complete cu_seqlens_q intervals" in str(error)
    else:
        raise AssertionError("partial TND logical batch must be rejected")


def test_pa_batch_random_context_uses_logical_batch_relations():
    q = torch.empty((4, 1, 1, 1), dtype=torch.float32)
    ori_kv = torch.empty((8, 2, 1, 1), dtype=torch.float32)
    fields = batch_kwargs(((0, 1, 1), (2, 3, 1)), 7001)
    fields.update({
        "layout_q": "BSND",
        "layout_kv": "PA_BBND",
        "seqused_q_values": [1, 1, 1, 1],
        "seqused_ori_kv_values": [4, 4, 4, 4],
    })

    with INPUTS_MODULE.TorchBatchRandomContext.from_case(q, ori_kv, None, fields):
        logical_kv = torch.rand(4, 1, 4, 2)

    assert torch.equal(logical_kv[0], logical_kv[2])


def test_tnd_pa_input_adapter_preserves_relation_through_block_tables():
    batch_size = 4
    q = torch.empty((batch_size, 64, 512), dtype=torch.bfloat16)
    ori_kv = torch.empty((batch_size, 128, 1, 512), dtype=torch.bfloat16)
    cmp_kv = torch.empty((batch_size, 1, 1, 512), dtype=torch.bfloat16)
    ori_block_table = torch.empty((batch_size, 1), dtype=torch.int32)
    cmp_block_table = torch.empty((batch_size, 1), dtype=torch.int32)
    cu_seqlens_q = torch.empty((batch_size + 1,), dtype=torch.int32)
    seqused_q = torch.empty((batch_size,), dtype=torch.int32)
    seqused_ori_kv = torch.empty((batch_size,), dtype=torch.int32)
    seqused_cmp_kv = torch.empty((batch_size,), dtype=torch.int32)
    cmp_residual_kv = torch.empty((batch_size,), dtype=torch.int32)
    sinks = torch.empty((64,), dtype=torch.float32)
    testcase_name = "SMLA_BATCH_TND_PA_INPUT_001"

    INPUTS_MODULE.generate_sparse_flash_mla_inputs(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        ori_block_table=ori_block_table,
        cmp_block_table=cmp_block_table,
        cu_seqlens_q=cu_seqlens_q,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        cmp_residual_kv=cmp_residual_kv,
        sinks=sinks,
        testcase_name=testcase_name,
        layout_q="TND",
        layout_kv="PA_BBND",
        S1=1,
        S2=128,
        softmax_scale=0.04419417,
        cmp_ratio=128,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        return_softmax_lse=True,
        has_ori_kv=True,
        has_cmp_kv=True,
        template_run_mode="HCA",
        cu_seqlens_q_values=[0, 1, 2, 3, 4],
        seqused_q_values=[1, 1, 1, 1],
        seqused_ori_kv_values=[128, 128, 128, 128],
        seqused_cmp_kv_values=[1, 1, 1, 1],
        cmp_residual_kv_values=[0, 0, 0, 0],
        input_ranges=((-10, 10), (-10, 10), (-10, 10)),
        **batch_kwargs(((0, 1, 1), (2, 3, 1)), 74123),
    )

    assert torch.equal(q[0], q[2])
    assert torch.equal(
        ori_kv[int(ori_block_table[0, 0])],
        ori_kv[int(ori_block_table[2, 0])],
    )
    assert torch.equal(
        cmp_kv[int(cmp_block_table[0, 0])],
        cmp_kv[int(cmp_block_table[2, 0])],
    )
    data = INPUTS_MODULE.INPUT_ADAPTER.load_golden_store().CASE_DATA.get(testcase_name)
    assert torch.equal(data["cpu_output"][0], data["cpu_output"][2])
    assert torch.equal(data["softmax_lse"][0], data["softmax_lse"][2])
