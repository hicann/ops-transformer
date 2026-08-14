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

"""White-box regression tests for the MQSMLA batch assets."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def load_asset_module(stem):
    path = Path(__file__).with_name("impl") / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"mqsmla_batch_test_{stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


COMPARE_MODULE = load_asset_module("compare")
INPUTS_MODULE = load_asset_module("inputs")


def batch_kwargs(slices, seed, sequence_slices=None):
    axes = (0,) if sequence_slices is None else (0, 1)
    slice_groups = (tuple(slices),)
    seed_groups = (tuple(seed for _ in slices),)
    ids = [
        tuple(
            f"{seed}_0_{len(range(start, stop, step))}_{step}"
            for start, stop, step in slices
        )
    ]
    if sequence_slices is not None:
        slice_groups += (tuple(sequence_slices),)
        seed_groups += (tuple(seed for _ in sequence_slices),)
        ids.append(
            tuple(
                f"{seed}_1_{len(range(start, stop, step))}_{step}"
                for start, stop, step in sequence_slices
            )
        )
    return {
        "batch_consistency_id": (tuple(ids),),
        "batch_axis": (axes,),
        "batch_slice_info": (slice_groups,),
        "batch_seed": (seed_groups,),
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


def test_same_case_batch_compare_maps_tnd_logical_sequence_slices():
    output = np.array(
        [[8.0], [9.0], [1.0], [2.0], [5.0], [6.0], [7.0], [1.0], [2.0]],
        dtype=np.float32,
    )
    fields = batch_kwargs(
        ((0, 1, 1), (1, 2, 1)),
        7001,
        ((2, 4, 1), (3, 5, 1)),
    )
    context = SimpleNamespace(
        attributes={
            "layout_q": "TND",
            "cu_seqlens_q_values": [0, 4, 9],
        }
    )

    result = COMPARE_MODULE.compare(
        output, output.copy(), compare_context=context, **fields
    )

    assert result[-1] == {"pass": True, "precision": "batch_intra=PASS"}


def test_batch_random_context_repeats_declared_batch_slices():
    q = torch.empty((3, 1, 1, 1), dtype=torch.float32)
    fields = batch_kwargs(((0, 1, 1), (2, 3, 1)), 7001)
    fields.update({"layout_q": "BSND", "layout_kv": "BSND"})
    original_uniform = np.random.uniform

    with INPUTS_MODULE.NumpyBatchRandomContext.from_case(q, None, None, fields):
        value = np.random.uniform(-1.0, 1.0, (3, 2))

    assert np.random.uniform is original_uniform
    assert np.array_equal(value[0], value[2])


def test_cross_case_changes_background_but_keeps_relation_slice():
    q = torch.empty((3, 2, 1, 1), dtype=torch.float32)
    fields = batch_kwargs(((0, 1, 1), (2, 3, 1)), 7001)
    fields.update({"layout_q": "BSND", "layout_kv": "BSND"})

    fields["testcase_name"] = "MQSMLA_CASE_A"
    with INPUTS_MODULE.NumpyBatchRandomContext.from_case(q, None, None, fields):
        first = np.random.uniform(-1.0, 1.0, q.shape)

    fields["testcase_name"] = "MQSMLA_CASE_B"
    with INPUTS_MODULE.NumpyBatchRandomContext.from_case(q, None, None, fields):
        second = np.random.uniform(-1.0, 1.0, q.shape)

    assert np.array_equal(first[0], first[2])
    assert np.array_equal(second[0], second[2])
    assert not np.array_equal(first[1], second[1])


def test_sparse_modes_map_randperm_and_accept_quant_mode_two():
    q = torch.empty((3, 2, 1, 1), dtype=torch.float32)
    fields = batch_kwargs(((0, 1, 1), (2, 3, 1)), 7001)
    fields.update(
        {
            "layout_q": "BSND",
            "layout_kv": "BSND",
            "seqused_q_values": [2, 1, 2],
        }
    )

    for mode in ("CSA", "ORI_SPARSE", "ORI_CMP_SPARSE"):
        context = INPUTS_MODULE.NumpyBatchRandomContext.from_case(q, None, None, fields)
        context.validate_params(
            {
                "template_run_mode": mode,
                "quant_mode": 1,
                "ori_sparse_indices_mode": "full",
                "cmp_sparse_indices_mode": "full",
                "ori_kv_topk_mode": "fullK",
                "cmp_kv_topk_mode": "fullK",
                "seqused_q": [2, 1, 2],
                "S1": 2,
                "N2": 1,
            }
        )
        with context:
            permutations = [torch.randperm(8) for _ in range(5)]
        assert torch.equal(permutations[0], permutations[3])
        assert torch.equal(permutations[1], permutations[4])

    quant_two = INPUTS_MODULE.NumpyBatchRandomContext.from_case(q, None, None, fields)
    quant_two.validate_params({"template_run_mode": "SWA", "quant_mode": 2})


def test_bsnd_batch_context_ignores_auxiliary_q_prefix():
    q = torch.empty((2, 1, 1, 1), dtype=torch.float32)
    fields = batch_kwargs(((0, 1, 1), (1, 2, 1)), 7001)
    fields.update(
        {
            "layout_q": "BSND",
            "layout_kv": "BSND",
            "cu_seqlens_q_values": [0, 14, 28],
        }
    )

    context = INPUTS_MODULE.NumpyBatchRandomContext.from_case(q, None, None, fields)

    assert context.q_prefix is None
    assert context.batch_size == 2


def test_tnd_batch_random_context_maps_logical_batch_and_token_ranges():
    q = torch.empty((8, 2), dtype=torch.float32)
    ori_kv = torch.empty((12, 1, 1), dtype=torch.float32)
    cmp_kv = torch.empty((4, 1, 1), dtype=torch.float32)
    fields = batch_kwargs(((0, 1, 1), (2, 3, 1)), 7001)
    fields.update(
        {
            "layout_q": "TND",
            "layout_kv": "TND",
            "cu_seqlens_q_values": [0, 2, 4, 6, 8],
            "cu_seqlens_ori_kv_values": [0, 3, 6, 9, 12],
            "cu_seqlens_cmp_kv_values": [0, 1, 2, 3, 4],
            "seqused_q_values": [2, 2, 2, 2],
            "seqused_ori_kv_values": [3, 3, 3, 3],
            "seqused_cmp_kv_values": [1, 1, 1, 1],
            "cmp_residual_kv_values": [0, 0, 0, 0],
        }
    )

    with INPUTS_MODULE.NumpyBatchRandomContext.from_case(q, ori_kv, cmp_kv, fields):
        q_value = np.random.uniform(-1.0, 1.0, (8, 2))
        ori_value = np.random.uniform(-1.0, 1.0, (12, 2))
        batch_value = np.random.uniform(-1.0, 1.0, (4, 2, 2))

    assert np.array_equal(q_value[0:2], q_value[4:6])
    assert np.array_equal(ori_value[0:3], ori_value[6:9])
    assert np.array_equal(batch_value[0], batch_value[2])


def test_tnd_sequence_relation_maps_logical_bs_to_physical_tokens():
    q = torch.empty((12, 2), dtype=torch.float32)
    ori_kv = torch.empty((15, 1, 1), dtype=torch.float32)
    fields = batch_kwargs(
        ((0, 1, 1), (2, 3, 1)),
        7001,
        ((1, 3, 1), (0, 2, 1)),
    )
    fields.update(
        {
            "layout_q": "TND",
            "layout_kv": "TND",
            "cu_seqlens_q_values": [0, 4, 8, 12],
            "cu_seqlens_ori_kv_values": [0, 5, 10, 15],
            "seqused_q_values": [4, 4, 4],
            "seqused_ori_kv_values": [5, 5, 5],
        }
    )

    with INPUTS_MODULE.NumpyBatchRandomContext.from_case(q, ori_kv, None, fields):
        q_value = np.random.uniform(-1.0, 1.0, (12, 2))
        ori_value = np.random.uniform(-1.0, 1.0, (15, 2))

    assert np.array_equal(q_value[1:3], q_value[8:10])
    assert np.array_equal(ori_value[0:5], ori_value[10:15])


def test_tnd_context_distinguishes_q_and_kv_with_the_same_token_extent():
    q = torch.empty((8, 2), dtype=torch.float32)
    ori_kv = torch.empty((8, 3), dtype=torch.float32)
    fields = batch_kwargs(((0, 1, 1), (2, 3, 1)), 7001)
    fields.update(
        {
            "layout_q": "TND",
            "layout_kv": "TND",
            "cu_seqlens_q_values": [0, 2, 4, 6, 8],
            "cu_seqlens_ori_kv_values": [0, 1, 4, 5, 8],
            "seqused_q_values": [2, 2, 2, 2],
            "seqused_ori_kv_values": [1, 3, 1, 3],
        }
    )

    with INPUTS_MODULE.NumpyBatchRandomContext.from_case(q, ori_kv, None, fields):
        q_value = np.random.uniform(-1.0, 1.0, q.shape)
        ori_value = np.random.uniform(-1.0, 1.0, ori_kv.shape)

    assert np.array_equal(q_value[0:2], q_value[4:6])
    assert np.array_equal(ori_value[0:1], ori_value[4:5])


def test_tnd_pa_input_adapter_preserves_relation_through_block_tables():
    batch_size = 4
    q = torch.empty((batch_size, 64, 512), dtype=torch.bfloat16)
    ori_kv = torch.empty((batch_size, 128, 1, 608), dtype=torch.float8_e4m3fn)
    cmp_kv = torch.empty((batch_size, 16, 1, 608), dtype=torch.float8_e4m3fn)
    ori_block_table = torch.empty((batch_size, 1), dtype=torch.int32)
    cmp_block_table = torch.empty((batch_size, 1), dtype=torch.int32)
    cu_seqlens_q = torch.empty((batch_size + 1,), dtype=torch.int32)
    seqused_q = torch.empty((batch_size,), dtype=torch.int32)
    seqused_ori_kv = torch.empty((batch_size,), dtype=torch.int32)
    seqused_cmp_kv = torch.empty((batch_size,), dtype=torch.int32)
    cmp_residual_kv = torch.empty((batch_size,), dtype=torch.int32)
    sinks = torch.empty((64,), dtype=torch.float32)

    INPUTS_MODULE.generate_mixed_quant_sparse_flash_mla_inputs(
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
        testcase_name="MQSMLA_BATCH_TND_PA_INPUT_001",
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
        quant_mode=1,
        rope_head_dim=64,
        return_softmax_lse=True,
        has_ori_kv=True,
        has_cmp_kv=True,
        template_run_mode="HCA",
        cu_seqlens_q_values=[0, 1, 2, 3, 4],
        seqused_q_values=[1, 1, 1, 1],
        seqused_ori_kv_values=[128, 128, 128, 128],
        seqused_cmp_kv_values=[1, 1, 1, 1],
        cmp_residual_kv_values=[0, 0, 0, 0],
        input_ranges=((-5, 5), (-5, 5), (-5, 5)),
        **batch_kwargs(((0, 1, 1), (2, 3, 1)), 74123),
    )

    assert torch.equal(q[0].view(torch.uint8), q[2].view(torch.uint8))
    assert torch.equal(
        ori_kv[int(ori_block_table[0, 0])].view(torch.uint8),
        ori_kv[int(ori_block_table[2, 0])].view(torch.uint8),
    )
    assert torch.equal(
        cmp_kv[int(cmp_block_table[0, 0])].view(torch.uint8),
        cmp_kv[int(cmp_block_table[2, 0])].view(torch.uint8),
    )
    assert torch.equal(ori_block_table[0], ori_block_table[2])
    assert torch.equal(cmp_block_table[0], cmp_block_table[2])


def test_non_batch_case_does_not_invent_pytest_geometry():
    q = torch.empty((1, 1, 64, 512), dtype=torch.bfloat16)
    ori_kv = torch.empty((1, 8, 1, 608), dtype=torch.float8_e4m3fn)
    cmp_kv = torch.empty((1, 2, 1, 608), dtype=torch.float8_e4m3fn)

    params = INPUTS_MODULE.INPUT_ADAPTER.build_case_params(
        q,
        ori_kv,
        cmp_kv,
        None,
        None,
        "BSND",
        "BSND",
        {},
    )

    assert params["block_size1"] is None
    assert params["block_size2"] is None
    assert "tile_size" not in params


def test_quant_mode_two_uses_pytest_logical_pa_key_without_assets_rebuild():
    q = torch.empty((1, 1, 1, 512), dtype=torch.bfloat16)
    ori_kv = torch.empty((1, 2, 1, 584), dtype=torch.float8_e4m3fn)
    ori_block_table = torch.empty((1, 1), dtype=torch.int32)
    seqused_q = torch.empty((1,), dtype=torch.int32)
    seqused_ori_kv = torch.empty((1,), dtype=torch.int32)
    sinks = torch.empty((1,), dtype=torch.float32)

    data = INPUTS_MODULE.generate_mixed_quant_sparse_flash_mla_inputs(
        q,
        ori_kv=ori_kv,
        ori_block_table=ori_block_table,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        sinks=sinks,
        testcase_name="MQSMLA_Q2_REPLAY_DECODER",
        layout_q="BSND",
        layout_kv="PA_BBND",
        S1=1,
        S2=2,
        block_num1=1,
        block_num2=0,
        block_size1=2,
        block_size2=2,
        quant_mode=2,
        tile_size=64,
        rope_head_dim=64,
        softmax_scale=0.04419417,
        cmp_ratio=1,
        ori_mask_mode=4,
        cmp_mask_mode=0,
        ori_win_left=127,
        ori_win_right=0,
        return_softmax_lse=False,
        has_ori_kv=True,
        has_cmp_kv=False,
        template_run_mode="SWA",
        cu_seqlens_q_values=[0, 1],
        seqused_q_values=[1],
        seqused_ori_kv_values=[2],
        cu_seqlens_ori_kv_values=[0, 2],
        input_ranges=((-1, 1), (-1, 1), None),
    )
    assert data["metadata_input"]["quant_mode"] == 2
    assert data["metadata_input"]["rope_head_dim"] == 64
    assert data["metadata_input"]["cu_seqlens_ori_kv"] is None
    assert data["metadata_input"]["cu_seqlens_cmp_kv"] is None
    assert data["metadata_input"]["ori_topk_length"] is None
    assert data["metadata_input"]["cmp_topk_length"] is None
    assert torch.equal(
        ori_kv.contiguous().view(torch.uint8),
        data["op_input"]["ori_kv"].contiguous().view(torch.uint8),
    )
    assert data["golden_state"]["ori_k_bnsd"] is not None


def test_batch_input_adapter_generates_equal_full_case_slices():
    batch_size = 2
    q = torch.empty((batch_size, 1, 64, 512), dtype=torch.bfloat16)
    ori_kv = torch.empty((batch_size, 8, 1, 608), dtype=torch.float8_e4m3fn)
    cmp_kv = torch.empty((batch_size, 2, 1, 608), dtype=torch.float8_e4m3fn)
    seqused_q = torch.empty((batch_size,), dtype=torch.int32)
    seqused_ori_kv = torch.empty((batch_size,), dtype=torch.int32)
    seqused_cmp_kv = torch.empty((batch_size,), dtype=torch.int32)
    cmp_residual_kv = torch.empty((batch_size,), dtype=torch.int32)
    sinks = torch.empty((64,), dtype=torch.float32)
    testcase_name = "MQSMLA_BATCH_FULL_INPUT_001"

    INPUTS_MODULE.generate_mixed_quant_sparse_flash_mla_inputs(
        q,
        ori_kv=ori_kv,
        cmp_kv=cmp_kv,
        seqused_q=seqused_q,
        seqused_ori_kv=seqused_ori_kv,
        seqused_cmp_kv=seqused_cmp_kv,
        cmp_residual_kv=cmp_residual_kv,
        sinks=sinks,
        testcase_name=testcase_name,
        layout_q="BSND",
        layout_kv="BSND",
        softmax_scale=0.04419417,
        cmp_ratio=4,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        quant_mode=1,
        rope_head_dim=64,
        return_softmax_lse=False,
        has_ori_kv=True,
        has_cmp_kv=True,
        seqused_q_values=[1, 1],
        seqused_ori_kv_values=[8, 8],
        seqused_cmp_kv_values=[2, 2],
        cmp_residual_kv_values=[0, 0],
        input_ranges=((-10, 10), (-5, 5), (-5, 5)),
        batch_axis=(
            (0,),
            (0,),
            (0,),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            (0,),
            (0,),
            (0,),
            (0,),
            None,
            None,
            None,
        ),
        batch_slice_info=(
            (((0, 1, 1), (1, 2, 1)),),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
        batch_seed=(
            ((74123, 74123),),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ),
    )

    def storage_equal(first, second):
        return torch.equal(
            first.detach().cpu().contiguous().view(torch.uint8),
            second.detach().cpu().contiguous().view(torch.uint8),
        )

    assert storage_equal(q[0], q[1])
    assert storage_equal(ori_kv[0], ori_kv[1])
    assert storage_equal(cmp_kv[0], cmp_kv[1])
    assert seqused_q.tolist() == [1, 1]
    assert seqused_ori_kv.tolist() == [8, 8]
    assert seqused_cmp_kv.tolist() == [2, 2]
    assert cmp_residual_kv.tolist() == [0, 0]
    data = INPUTS_MODULE.INPUT_ADAPTER.load_golden_store().CASE_DATA.get(testcase_name)
    assert data is not None
    assert data["op_input"]["cu_seqlens_q"] is None
    assert data["op_input"]["cu_seqlens_ori_kv"] is None
    assert data["op_input"]["cu_seqlens_cmp_kv"] is None
    assert data["metadata_input"]["cu_seqlens_ori_kv"] is None
    assert data["metadata_input"]["cu_seqlens_cmp_kv"] is None
    assert data["metadata_input"]["ori_topk_length"] is None
    assert data["metadata_input"]["cmp_topk_length"] is None
    assert data["cpu_output"] is None
    INPUTS_MODULE.INPUT_ADAPTER.load_golden_store().materialize_golden(data)
    assert data["cpu_output"] is not None
