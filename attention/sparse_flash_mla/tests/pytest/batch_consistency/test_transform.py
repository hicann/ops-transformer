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

import pytest
import torch

from .adapter import CaseAdapter
from .schema import SchemaError, check_invariants
from .transform import (
    ActualInputSemanticOracle,
    BatchCaseTransformer,
    InvalidTransformError,
)


def make_padded_bsnd_case():
    return {
        "B": 3,
        "layout_q": "BSND",
        "layout_kv": "BSND",
        "params": {},
        "cpu_output": torch.zeros((3, 2, 1, 2)),
        "metadata_input": {
            "B": 3,
            "N1": 1,
            "N2": 1,
            "D": 2,
            "max_seqlen_q": 2,
            "max_seqlen_ori_kv": 8,
            "max_seqlen_cmp_kv": 2,
            "has_ori_kv": True,
            "has_cmp_kv": True,
            "layout_q": "BSND",
            "layout_kv": "BSND",
        },
        "input": {
            "q": torch.arange(12).reshape(3, 2, 1, 2),
            "ori_kv": torch.arange(48).reshape(3, 8, 1, 2),
            "cmp_kv": torch.arange(12).reshape(3, 2, 1, 2),
            "cu_seqlens_q": None,
            "seqused_ori_kv": torch.tensor([4, 4, 8], dtype=torch.int32),
            "seqused_cmp_kv": torch.tensor([0, 0, 2], dtype=torch.int32),
            "cmp_residual_kv": torch.tensor([0, 0, 0], dtype=torch.int32),
            "sinks": torch.zeros(1),
            "softmax_scale": 1.0,
            "cmp_ratio": 4,
            "ori_mask_mode": 4,
            "cmp_mask_mode": 3,
            "ori_win_left": -1,
            "ori_win_right": -1,
            "layout_q": "BSND",
            "layout_kv": "BSND",
        },
    }


@pytest.mark.parametrize("batch_ids", [[0], [0, 1], [1, 0], [2, 0]])
def test_bsnd_batch_selection_keeps_padded_kv_metadata(batch_ids):
    baseline = make_padded_bsnd_case()
    derived = BatchCaseTransformer(baseline).select_batches(batch_ids)
    for prefix, physical_length in (("ori", 8), ("cmp", 2)):
        assert derived["metadata_input"][f"max_seqlen_{prefix}_kv"] == physical_length
        assert derived["input"][f"{prefix}_kv"].shape[1] == physical_length
        assert torch.equal(
            derived["input"][f"seqused_{prefix}_kv"],
            baseline["input"][f"seqused_{prefix}_kv"][batch_ids],
        )
    assert derived["metadata_input"]["has_cmp_kv"] is True
    mapping = [
        (old_batch, token, new_batch, token)
        for new_batch, old_batch in enumerate(batch_ids)
        for token in range(2)
    ]
    assert ActualInputSemanticOracle.validate_mapped_tokens(baseline, derived, mapping)[
        "pass"
    ]


@pytest.mark.parametrize("prefix", ["ori", "cmp"])
@pytest.mark.parametrize("configured", [0, 1, 9])
def test_bsnd_invariants_reject_metadata_different_from_storage(prefix, configured):
    data = make_padded_bsnd_case()
    data["metadata_input"][f"max_seqlen_{prefix}_kv"] = configured
    with pytest.raises(SchemaError, match=f"BSND metadata max_seqlen_{prefix}_kv"):
        check_invariants(data)


def test_bsnd_logical_kv_resize_keeps_padded_metadata():
    data = make_padded_bsnd_case()
    transformer = BatchCaseTransformer(data)
    transformer._set_kv_lengths(data, "ori", [0, 0, 0])
    transformer._set_kv_lengths(data, "cmp", [0, 0, 0])
    transformer._sync_derived_fields(CaseAdapter(data))
    check_invariants(data)
    assert data["metadata_input"]["max_seqlen_ori_kv"] == 8
    assert data["metadata_input"]["max_seqlen_cmp_kv"] == 2
    assert data["input"]["seqused_ori_kv"].tolist() == [0, 0, 0]
    assert data["input"]["seqused_cmp_kv"].tolist() == [0, 0, 0]


def make_kv_resize_case(layout, batch_size=1, prefix="ori"):
    shapes = {
        "TND": (batch_size * 4, 1, 2),
        "BSND": (batch_size, 4, 1, 2),
        "PA_BBND": (batch_size, 4, 1, 2),
    }
    return {
        "params": {},
        "metadata_input": {"batch_size": batch_size},
        "op_input": {
            "layout_kv": layout,
            f"{prefix}_kv": torch.arange(batch_size * 8).reshape(shapes[layout]),
            f"cu_seqlens_{prefix}_kv": torch.arange(batch_size + 1, dtype=torch.int32)
            * 4,
            f"seqused_{prefix}_kv": torch.full((batch_size,), 4, dtype=torch.int32),
            f"{prefix}_block_table": torch.arange(
                batch_size, dtype=torch.int32
            ).reshape(batch_size, 1),
        },
    }


@pytest.mark.parametrize("prefix", ["ori", "cmp"])
def test_tnd_resize_rejects_zero_physical_kv_without_mutation(prefix):
    data = make_kv_resize_case("TND", prefix=prefix)
    before = dict(data["op_input"])
    transformer = object.__new__(BatchCaseTransformer)
    with pytest.raises(InvalidTransformError, match="zero physical token dimension"):
        transformer._set_kv_lengths(data, prefix, [0])
    assert all(data["op_input"][key] is value for key, value in before.items())


def test_tnd_resize_keeps_mixed_zero_and_nonzero_lengths():
    data = make_kv_resize_case("TND", batch_size=2)
    original = data["op_input"]["ori_kv"].clone()
    object.__new__(BatchCaseTransformer)._set_kv_lengths(data, "ori", [0, 3])
    tensors = data["op_input"]
    assert torch.equal(tensors["ori_kv"], original[4:7])
    assert tensors["cu_seqlens_ori_kv"].tolist() == [0, 0, 3]
    assert tensors["seqused_ori_kv"].tolist() == [0, 3]


@pytest.mark.parametrize("layout", ["BSND", "PA_BBND"])
def test_zero_logical_kv_preserves_nonzero_physical_storage(layout):
    data = make_kv_resize_case(layout)
    original = data["op_input"]["ori_kv"].clone()
    object.__new__(BatchCaseTransformer)._set_kv_lengths(data, "ori", [0])
    assert torch.equal(data["op_input"]["ori_kv"], original)
    assert data["op_input"]["seqused_ori_kv"].tolist() == [0]


def test_absent_optional_cmp_remains_none():
    data = make_kv_resize_case("TND", prefix="cmp")
    data["op_input"]["cmp_kv"] = None
    object.__new__(BatchCaseTransformer)._set_kv_lengths(data, "cmp", [0])
    assert data["op_input"]["cmp_kv"] is None


def make_unbounded_swa_case(q_length, ori_length):
    return {
        "params": {},
        "metadata_input": {"batch_size": 1},
        "op_input": {
            "q": torch.zeros((q_length, 1, 1)),
            "q_descale": None,
            "cu_seqlens_q": torch.tensor([0, q_length], dtype=torch.int32),
            "seqused_q": torch.tensor([q_length], dtype=torch.int32),
            "seqused_ori_kv": torch.tensor([ori_length], dtype=torch.int32),
            "seqused_cmp_kv": None,
            "cmp_kv": None,
            "sinks": torch.zeros((1,)),
            "ori_mask_mode": 4,
            "cmp_mask_mode": 0,
            "ori_win_left": -1,
            "ori_win_right": -1,
            "layout_q": "TND",
            "layout_kv": "PA_BBND",
        },
    }


def test_unbounded_mode4_rejects_token_split_that_shortens_kv():
    baseline = make_unbounded_swa_case(q_length=2, ori_length=4096)
    derived = make_unbounded_swa_case(q_length=1, ori_length=4095)

    assert ActualInputSemanticOracle._windows(baseline, 0, 0)["ori"] == (0, 4096)
    assert ActualInputSemanticOracle._windows(derived, 0, 0)["ori"] == (0, 4095)
    with pytest.raises(InvalidTransformError, match="different mask window"):
        ActualInputSemanticOracle.validate_mapped_tokens(
            baseline, derived, [(0, 0, 0, 0)]
        )


@pytest.mark.parametrize("layout", ["TND", "BSND"])
@pytest.mark.parametrize("with_lengths", [False, True])
@pytest.mark.parametrize("prefix", ["ori", "cmp"])
def test_shape_resize_copies_valid_sparse_rows(layout, with_lengths, prefix):
    data = make_unbounded_swa_case(2, 4)
    tensors = data["op_input"]
    tensors["layout_q"] = layout
    tensors["layout_kv"] = "BSND"
    tensors["ori_kv"] = torch.zeros((1, 4, 1, 1))
    tensors[f"{prefix}_kv"] = torch.zeros((1, 4, 1, 1))
    tensors[f"seqused_{prefix}_kv"] = torch.tensor([4], dtype=torch.int32)
    tensors[f"{prefix}_mask_mode"] = 0
    rows = [[0, 1, 2], [1, 2, -1 if with_lengths else 3]]
    tensors[f"{prefix}_sparse_indices"] = torch.tensor(rows).reshape(2, 1, 3)
    if with_lengths:
        tensors[f"{prefix}_topk_length"] = torch.tensor([[3], [2]])
    if layout == "BSND":
        tensors["q"] = tensors["q"].unsqueeze(0)
        for name in (f"{prefix}_sparse_indices", f"{prefix}_topk_length"):
            if name in tensors:
                tensors[name] = tensors[name].unsqueeze(0)
    original = tensors[f"{prefix}_sparse_indices"].clone()
    transformer = object.__new__(BatchCaseTransformer)
    transformer.layout_q = layout
    transformer.tensors = dict(tensors)
    transformer._resize_query_prefix(data, 4)
    axis = 1 if layout == "BSND" else 0
    resized = tensors[f"{prefix}_sparse_indices"]
    assert torch.equal(resized.narrow(axis, 0, 2), original)
    assert torch.equal(resized.select(axis, 2), original.select(axis, 1))
    assert torch.equal(resized.select(axis, 3), original.select(axis, 1))
    transformer._validate_shape_change_sparse_indices(data)
    resized.select(axis, 3)[..., 0] = -1
    with pytest.raises(InvalidTransformError, match="effective top-k range"):
        transformer._validate_shape_change_sparse_indices(data)
