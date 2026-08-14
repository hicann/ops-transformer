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
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def load_asset_module(stem):
    path = Path(__file__).with_name("impl") / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(f"smla_batch_test_{stem}", path)
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
    original_rand = torch.rand

    with INPUTS_MODULE.TorchBatchRandomContext.from_case(q, None, None, fields):
        value = torch.rand(3, 2)

    assert torch.rand is original_rand
    assert torch.equal(value[0], value[2])


def test_cross_case_changes_background_but_keeps_relation_slice():
    q = torch.empty((3, 2, 1, 1), dtype=torch.float32)
    fields = batch_kwargs(((0, 1, 1), (2, 3, 1)), 7001)
    fields.update({"layout_q": "BSND", "layout_kv": "BSND"})

    fields["testcase_name"] = "SMLA_CASE_A"
    with INPUTS_MODULE.TorchBatchRandomContext.from_case(q, None, None, fields):
        first = torch.rand(q.shape)

    fields["testcase_name"] = "SMLA_CASE_B"
    with INPUTS_MODULE.TorchBatchRandomContext.from_case(q, None, None, fields):
        second = torch.rand(q.shape)

    assert torch.equal(first[0], first[2])
    assert torch.equal(second[0], second[2])
    assert not torch.equal(first[1], second[1])


def test_sparse_modes_map_randperm_by_relation_with_nonuniform_batches():
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
        context = INPUTS_MODULE.TorchBatchRandomContext.from_case(q, None, None, fields)
        context.validate_params(
            {
                "template_mode": mode,
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

    context = INPUTS_MODULE.TorchBatchRandomContext.from_case(q, None, None, fields)

    assert context.q_prefix is None
    assert context.batch_size == 2


def test_bsnd_input_adapter_rejects_prefix_tensor_slot(monkeypatch):
    q = torch.empty((2, 1, 1, 1), dtype=torch.float32)
    ori_kv = torch.empty((2, 1, 1, 1), dtype=torch.float32)
    cmp_kv = torch.empty((2, 1, 1, 1), dtype=torch.float32)
    prefix_slot = torch.empty((3,), dtype=torch.int32)
    generated = {
        "input": {
            "q": q.clone(),
            "ori_kv": ori_kv.clone(),
            "cmp_kv": cmp_kv.clone(),
        },
        "metadata_input": {},
    }
    monkeypatch.setattr(
        INPUTS_MODULE.INPUT_ADAPTER,
        "customize",
        lambda *_args, **_kwargs: generated,
    )

    with pytest.raises(ValueError, match="cu_seqlens_q is present in CSV"):
        INPUTS_MODULE.generate_sparse_flash_mla_inputs(
            q,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            cu_seqlens_q=prefix_slot,
            layout_q="BSND",
            layout_kv="BSND",
        )


def test_topk_override_preserves_csv_values_and_masked_legacy_slots():
    tensor = torch.full((2, 1), 7, dtype=torch.int32)
    explicit = [[3], [4]]
    generated_context = SimpleNamespace(attributes={})
    explicit_context = SimpleNamespace(attributes={"ori_topk_length": explicit})

    assert (
        INPUTS_MODULE.INPUT_ADAPTER.select_topk_override(
            "ori_topk_length", tensor, 0, None
        )
        is tensor
    )
    assert (
        INPUTS_MODULE.INPUT_ADAPTER.select_topk_override(
            "ori_topk_length", tensor, 0, generated_context
        )
        is None
    )
    assert (
        INPUTS_MODULE.INPUT_ADAPTER.select_topk_override(
            "ori_topk_length", tensor, 0, explicit_context
        )
        == explicit
    )
    assert (
        INPUTS_MODULE.INPUT_ADAPTER.select_topk_override(
            "ori_topk_length", tensor, 3, None
        )
        is tensor
    )


def test_mask_zero_topk_is_generated_then_copied_to_ttk_tensor(monkeypatch):
    q = torch.zeros((1, 2, 1, 4), dtype=torch.float32)
    ori_kv = torch.zeros((1, 4, 1, 4), dtype=torch.float32)
    ori_sparse_indices = torch.zeros((1, 2, 1, 4), dtype=torch.int32)
    ori_topk_length = torch.full((1, 2, 1), 99, dtype=torch.int32)
    generated_topk = torch.tensor([[[4], [3]]], dtype=torch.int32)
    captured = {}

    def capture_customize(*args):
        captured["params"] = args[7]
        return {
            "input": {
                "q": q.clone(),
                "ori_kv": ori_kv.clone(),
                "ori_sparse_indices": ori_sparse_indices.clone(),
                "ori_topk_length": generated_topk,
            },
            "metadata_input": {"ori_topk_length": generated_topk},
        }

    monkeypatch.setattr(INPUTS_MODULE.INPUT_ADAPTER, "customize", capture_customize)

    INPUTS_MODULE.generate_sparse_flash_mla_inputs(
        q,
        ori_kv=ori_kv,
        ori_sparse_indices=ori_sparse_indices,
        ori_topk_length=ori_topk_length,
        ori_mask_mode=0,
        layout_q="BSND",
        layout_kv="BSND",
        context=SimpleNamespace(attributes={}),
    )

    assert captured["params"]["ori_topk_length"] is None
    assert torch.equal(ori_topk_length, generated_topk)


def test_tnd_batch_random_context_maps_q_and_kv_token_ranges():
    q = torch.empty((8, 1, 1), dtype=torch.float32)
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

    with INPUTS_MODULE.TorchBatchRandomContext.from_case(q, ori_kv, cmp_kv, fields):
        q_value = torch.rand(q.shape)
        ori_value = torch.rand(ori_kv.shape)
        cmp_value = torch.rand(cmp_kv.shape)
        batch_value = torch.rand(4, 2, 2)

    assert torch.equal(q_value[0:2], q_value[4:6])
    assert torch.equal(ori_value[0:3], ori_value[6:9])
    assert torch.equal(cmp_value[0:1], cmp_value[2:3])
    assert torch.equal(batch_value[0], batch_value[2])


def test_tnd_batch_random_context_distinguishes_equal_q_and_kv_extents():
    q = torch.empty((8, 2, 1), dtype=torch.float32)
    ori_kv = torch.empty((8, 1, 1), dtype=torch.float32)
    fields = batch_kwargs(
        ((0, 1, 1), (2, 3, 1)),
        7001,
        ((0, 1, 1), (1, 2, 1)),
    )
    fields.update(
        {
            "layout_q": "TND",
            "layout_kv": "TND",
            "cu_seqlens_q_values": [0, 2, 4, 6, 8],
            "cu_seqlens_ori_kv_values": [0, 2, 4, 6, 8],
            "seqused_q_values": [2, 2, 2, 2],
            "seqused_ori_kv_values": [2, 2, 2, 2],
        }
    )

    with INPUTS_MODULE.TorchBatchRandomContext.from_case(q, ori_kv, None, fields):
        q_value = torch.rand(8, 2, 1)
        ori_value = torch.rand(8, 1, 1)

    assert torch.equal(q_value[0:1], q_value[5:6])
    assert torch.equal(ori_value[0:2], ori_value[4:6])


def test_tnd_batch_random_context_rejects_out_of_range_logical_batch():
    q = torch.empty((8, 1, 1), dtype=torch.float32)
    fields = batch_kwargs(((1, 3, 1), (3, 5, 1)), 7001)
    fields.update(
        {
            "layout_q": "TND",
            "layout_kv": "PA_BBND",
            "cu_seqlens_q_values": [0, 2, 4, 6, 8],
        }
    )

    try:
        INPUTS_MODULE.TorchBatchRandomContext.from_case(q, None, None, fields)
    except ValueError as error:
        assert "logical B slice must be in-range" in str(error)
    else:
        raise AssertionError("out-of-range logical batch must be rejected")


def test_tnd_sequence_relation_maps_logical_bs_to_physical_tokens():
    q = torch.empty((12, 1, 1), dtype=torch.float32)
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

    with INPUTS_MODULE.TorchBatchRandomContext.from_case(q, ori_kv, None, fields):
        q_value = torch.rand(q.shape)
        ori_value = torch.rand(ori_kv.shape)

    assert torch.equal(q_value[1:3], q_value[8:10])
    assert torch.equal(ori_value[0:5], ori_value[10:15])


def test_sequence_relation_different_positions_requires_no_mask():
    q = torch.empty((8, 1, 1), dtype=torch.float32)
    fields = batch_kwargs(
        ((0, 1, 1), (1, 2, 1)),
        7001,
        ((0, 2, 1), (2, 4, 1)),
    )
    fields.update(
        {
            "layout_q": "TND",
            "layout_kv": "TND",
            "cu_seqlens_q_values": [0, 4, 8],
            "seqused_q_values": [4, 4],
        }
    )
    context = INPUTS_MODULE.TorchBatchRandomContext.from_case(q, None, None, fields)

    try:
        context.validate_params({"template_mode": "HCA", "ori_mask_mode": 4})
    except ValueError as error:
        assert "require no-mask mode" in str(error)
    else:
        raise AssertionError(
            "position-dependent mask must reject shifted logical S slices"
        )


def test_pa_batch_random_context_uses_logical_batch_relations():
    q = torch.empty((4, 1, 1, 1), dtype=torch.float32)
    ori_kv = torch.empty((8, 2, 1, 1), dtype=torch.float32)
    fields = batch_kwargs(((0, 1, 1), (2, 3, 1)), 7001)
    fields.update(
        {
            "layout_q": "BSND",
            "layout_kv": "PA_BBND",
            "seqused_q_values": [1, 1, 1, 1],
            "seqused_ori_kv_values": [4, 4, 4, 4],
        }
    )

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
    assert torch.equal(ori_block_table[0], ori_block_table[2])
    assert torch.equal(cmp_block_table[0], cmp_block_table[2])
    assert torch.equal(
        ori_kv[int(ori_block_table[0, 0])],
        ori_kv[int(ori_block_table[2, 0])],
    )
    assert torch.equal(
        cmp_kv[int(cmp_block_table[0, 0])],
        cmp_kv[int(cmp_block_table[2, 0])],
    )
    data = INPUTS_MODULE.INPUT_ADAPTER.load_golden_store().CASE_DATA.get(testcase_name)
    assert data["cpu_output"] is None
    assert data["softmax_lse"] is None
    INPUTS_MODULE.INPUT_ADAPTER.load_golden_store().materialize_golden(data)
    assert torch.equal(data["cpu_output"][0], data["cpu_output"][2])
    assert data["softmax_lse"] is not None
