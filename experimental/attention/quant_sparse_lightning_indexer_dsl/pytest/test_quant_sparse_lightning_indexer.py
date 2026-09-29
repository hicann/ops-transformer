# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from quant_sparse_lightning_indexer_golden import pack_candidate_k
import torch
import torch_npu


TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIR))

from qsli_test_utils import result_compare_method
from quant_sparse_lightning_indexer_golden import (  # noqa: E402
    pack_mxfp4,
    quant_sparse_lightning_indexer_golden,
)
import cann_ops_transformer.ops.attention.quant_sparse_lightning_indexer_dsl  # noqa: F401,E402


quant_sparse_lightning_indexer = (
    torch.ops.cann_ops_transformer.ds41.quant_sparse_lightning_indexer
)


def _case(batch: int, s2: int, seed: int) -> dict[str, torch.Tensor]:
    """One TND query per batch with shuffled PA pages and candidates."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    pages_per_batch = s2 // 128
    physical_pages = batch * pages_per_batch
    q = pack_mxfp4(torch.randn((batch, 32, 128), generator=generator) * 0.25)
    k = pack_mxfp4(
        torch.randn((physical_pages, 128, 1, 128), generator=generator) * 0.25
    )
    candidate = torch.full((batch, 1, 2048), -1, dtype=torch.int32)
    block_table = torch.empty((batch, pages_per_batch), dtype=torch.int32)
    logical_candidate_blocks = s2 // 8
    candidate_blocks = min(logical_candidate_blocks, 2048)
    for batch_idx in range(batch):
        page_begin = batch_idx * pages_per_batch
        block_table[batch_idx] = (
            torch.randperm(pages_per_batch, generator=generator).to(torch.int32)
            + page_begin
        )
        candidate[batch_idx, 0, :candidate_blocks] = torch.randperm(
            logical_candidate_blocks, generator=generator
        )[:candidate_blocks].to(torch.int32)
    return {
        "q": q,
        "k": k,
        "w": torch.randn((batch, 32), generator=generator),
        "descale_q": torch.randint(
            125, 130, (batch, 32, 2, 2), generator=generator, dtype=torch.uint8
        ),
        "descale_k": torch.randint(
            125,
            130,
            (physical_pages, 128, 1, 2, 2),
            generator=generator,
            dtype=torch.uint8,
        ),
        "candidate": candidate,
        "candidate_length": torch.full((batch, 1), candidate_blocks, dtype=torch.int32),
        "cu": torch.arange(batch + 1, dtype=torch.int32),
        "used_k": torch.full((batch,), s2, dtype=torch.int32),
        "block_table": block_table,
        "offset": (
            torch.arange(batch, dtype=torch.int32).reshape(batch, 1) * 10000 + 37
        ),
    }


def _remove_output_offset(indices: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    logical = indices.clone()
    expanded_offsets = offsets.reshape(indices.shape[0], 1, 1).expand_as(indices)
    valid = logical >= 0
    logical[valid] -= expanded_offsets[valid]
    return logical


def _asc_compare_params(case: dict[str, torch.Tensor], topk: int) -> tuple:
    batch = int(case["used_k"].numel())
    q_lengths = torch.diff(case["cu"]).tolist()
    return (
        batch,
        max(q_lengths),
        int(case["used_k"].max()),
        int(case["q"].shape[0]),
        int(case["k"].shape[0] * 128),
        32,
        1,
        128,
        128,
        int(case["k"].shape[0]),
        torch.uint8,
        torch.float32,
        torch.uint8,
        torch.int32,
        case["cu"].tolist(),
        case["used_k"].tolist(),
        1,
        1,
        "TND",
        "PA_BBND",
        topk,
        0,
        (-1, 1),
        (-1, 1),
        (-1, 1),
        (125, 129),
        (125, 129),
    )


def _check_sparse_values(
    actual_indices: torch.Tensor,
    actual_values: torch.Tensor,
    golden: dict[str, torch.Tensor],
) -> None:
    logical = _remove_output_offset(actual_indices, golden["_output_idx_offset"])
    cu = golden["_cu_seqlens_q"].tolist()
    reference = torch.zeros_like(actual_values, dtype=torch.float32)
    for batch, (begin, end) in enumerate(zip(cu, cu[1:])):
        for row in range(begin, end):
            valid = logical[row, 0] >= 0
            reference[row, 0, valid] = golden["_topk_value"][
                batch, 0, row - begin, logical[row, 0, valid].long()
            ]
    actual_np = actual_values.to(torch.float32).numpy().reshape(-1)
    reference_np = reference.numpy().reshape(-1)
    assert result_compare_method.judge_value_by_isclose(actual_np, reference_np), (
        "sparse_values failed the local ASC result comparison"
    )


def _run_and_compare(case: dict[str, torch.Tensor], topk: int) -> None:
    golden = quant_sparse_lightning_indexer_golden(
        case["q"],
        pack_candidate_k(case["k"], case["descale_k"]),
        case["w"],
        case["descale_q"],
        case["candidate"],
        case["candidate_length"],
        cu_seqlens_q=case["cu"],
        seqused_k=case["used_k"],
        block_table=case["block_table"],
        output_idx_offset=case["offset"],
        topk=topk,
        candidate_block_size=8,
        return_value=True,
    )
    actual_outputs = quant_sparse_lightning_indexer(
        case["q"].npu(),
        pack_candidate_k(case["k"], case["descale_k"]).npu(),
        case["w"].npu(),
        case["descale_q"].npu(),
        case["candidate"].npu(),
        case["candidate_length"].npu(),
        topk,
        1,
        8,
        cu_seqlens_q=case["cu"].npu(),
        seqused_k=case["used_k"].npu(),
        block_table=case["block_table"].npu(),
        output_idx_offset=case["offset"].npu(),
        return_value=True,
        layout_k="PA_BBND",
    )
    torch.npu.synchronize()
    actual = {
        name: value.cpu()
        for name, value in zip(("sparse_indices", "sparse_values"), actual_outputs)
    }

    expected_indices = _remove_output_offset(
        golden["sparse_indices"], golden["_output_idx_offset"]
    )
    actual_indices = _remove_output_offset(
        actual["sparse_indices"], golden["_output_idx_offset"]
    )
    result, fulfill_percent = result_compare_method.check_result(
        expected_indices,
        actual_indices,
        golden["_topk_value"],
        _asc_compare_params(case, topk),
    )
    assert result == "Pass", f"TopK compare failed: {fulfill_percent:.6f}%"
    _check_sparse_values(actual["sparse_indices"], actual["sparse_values"], golden)


@pytest.mark.npu
def test_qsli_shuffled_pa_candidate_scale_and_offset_matches_golden():
    _run_and_compare(_case(batch=2, s2=256, seed=221), topk=64)


@pytest.mark.npu
@pytest.mark.slow
def test_qsli_batch8_sequence8k_matches_golden():
    """Target acceptance shape: B=8, S2=8192, C=2048, valid C=1024."""
    _run_and_compare(_case(batch=8, s2=8192, seed=8192), topk=512)


@pytest.mark.npu
@pytest.mark.slow
def test_qsli_batch1_sequence128k_matches_golden():
    """Long-PA acceptance shape: B=1, S2=131072, C=2048, topk=512."""
    _run_and_compare(_case(batch=1, s2=128 * 1024, seed=131072), topk=512)


@pytest.mark.npu
def test_candidate_block_length_is_exact_prefix_count():
    case = _case(batch=1, s2=256, seed=9)
    case["candidate_length"][0, 0] = 17
    case["candidate"][0, 0, 17:] = 2**30
    _run_and_compare(case, topk=64)


def _compare_structure(expected, actual, layout="TND", tied=False):
    expected = torch.tensor(expected, dtype=torch.int32).reshape(1, 1, 4)
    actual = torch.tensor(actual, dtype=torch.int32).reshape(1, 1, 4)
    if layout == "BSND":
        expected = expected.reshape(1, 1, 1, 4)
        actual = actual.reshape(1, 1, 1, 4)
    params = (
        1,
        1,
        8,
        1,
        8,
        32,
        1,
        128,
        128,
        1,
        torch.uint8,
        torch.float32,
        torch.uint8,
        torch.int32,
        [0, 1],
        [8],
        1,
        1,
        layout,
        "PA_BBND",
        4,
        0,
        (-1, 1),
        (-1, 1),
        (0, 1),
        (125, 129),
        (125, 129),
    )
    scores = (
        np.ones((1, 1, 1, 8), dtype=np.float32)
        if tied
        else np.arange(8, 0, -1, dtype=np.float32).reshape(1, 1, 1, 8)
    )
    return result_compare_method.check_result(expected, actual, scores, params)[0]


@pytest.mark.parametrize("layout", ["TND", "BSND"])
@pytest.mark.parametrize(
    "expected,actual",
    [
        ([-1, -1, -1, -1], [0, 0, 0, 0]),
        ([-1, -1, -1, -1], [-1, -1, -1, 0]),
        ([-1, -1, -1, -1], [-2, -1, -1, -1]),
        ([0, 1, -1, -1], [0, 1, 2, -1]),
        ([0, 1, -1, -1], [0, -1, 1, -1]),
        ([0, 1, 2, 3], [0, 1, 2, -1]),
        ([0, 1, 2, 3], [0, 1, 2, 2]),
        ([0, 1, -1, -1], [0, 0, -1, -1]),
    ],
)
def test_reject_invalid_structure(expected, actual, layout):
    assert _compare_structure(expected, actual, layout) == "Failed"


@pytest.mark.parametrize("layout", ["TND", "BSND"])
@pytest.mark.parametrize(
    "expected,actual",
    [
        ([-1, -1, -1, -1], [-1, -1, -1, -1]),
        ([0, 1, -1, -1], [1, 0, -1, -1]),
        ([0, 1, 2, 3], [3, 1, 0, 2]),
    ],
)
def test_accept_valid_structure(expected, actual, layout):
    assert _compare_structure(expected, actual, layout) == "Pass"


def test_allow_distinct_indices_with_equal_scores():
    assert _compare_structure([0, 1, 2, 3], [4, 5, 6, 7], tied=True) == "Pass"


@pytest.mark.parametrize("return_value", [False, True])
def test_close_integer_indices_cannot_rescue_wrong_scores(return_value):
    compare_topk_valid = result_compare_method.compare_topk_valid
    scores = np.zeros((1, 1, 1, 1002), dtype=np.float32)
    scores[0, 0, 0, 1000] = 10.0
    scores[0, 0, 0, 1001] = 1.0
    passed, _ = compare_topk_valid(
        [1000],
        [1001],
        scores,
        (0, 0, 0),
        [],
        [],
        np.array([1.0]),
        np.array([10.0]),
        thres=0.001,
        return_value_flag=return_value,
    )
    assert not passed


def test_boundary_tolerance_matches_qli():
    compare_topk_valid = result_compare_method.compare_topk_valid
    scores = np.array([[[[1.0, 0.9995]]]], dtype=np.float32)
    assert compare_topk_valid(
        [0], [1], scores, (0, 0, 0), [], [], thres=0.001, return_value_flag=False
    )[0]


@pytest.mark.parametrize("enabled", [False, True])
def test_value_fallback_respects_return_value(enabled):
    compare_topk_valid = result_compare_method.compare_topk_valid
    scores = np.array([[[[1.0, 0.9]]]], dtype=np.float32)
    passed, _ = compare_topk_valid(
        [0],
        [1],
        scores,
        (0, 0, 0),
        [],
        [],
        np.array([1.0]),
        np.array([1.0]),
        thres=0.001,
        return_value_flag=enabled,
    )
    assert passed == enabled


@pytest.mark.parametrize(
    "enabled,actual_value,expected_pass",
    [
        (False, 10.0, False),
        (True, 1.0, False),
        (True, 10.0, True),
    ],
)
def test_entrypoint_uses_real_values_not_integer_indices(
    enabled, actual_value, expected_pass
):
    expected = torch.tensor([[[1000, -1, -1, -1]]], dtype=torch.int32)
    actual = torch.tensor([[[1001, -1, -1, -1]]], dtype=torch.int32)
    params = (
        1,
        1,
        1002,
        1,
        1002,
        32,
        1,
        128,
        128,
        8,
        torch.uint8,
        torch.float32,
        torch.uint8,
        torch.int32,
        [0, 1],
        [1002],
        1,
        1,
        "TND",
        "PA_BBND",
        4,
        0,
        (-1, 1),
        (-1, 1),
        (0, 1),
        (125, 129),
        (125, 129),
    )
    scores = np.zeros((1, 1, 1, 1002), dtype=np.float32)
    scores[0, 0, 0, 1000] = 10.0
    scores[0, 0, 0, 1001] = 1.0
    result, _ = result_compare_method.check_result(
        expected,
        actual,
        scores,
        params,
        return_value=enabled,
        cpu_topk_value=torch.tensor([[[10.0, 0, 0, 0]]]),
        npu_topk_value=torch.tensor([[[actual_value, 0, 0, 0]]]),
    )
    assert (result == "Pass") == expected_pass


@pytest.mark.parametrize(
    "force_bf16,real,expected_pass",
    [
        (False, 1.004, True),
        (False, 1.006, False),
        (True, 1.006, True),
        (True, 1.009, False),
    ],
)
def test_value_tolerance_matches_asc(force_bf16, real, expected_pass):
    judge_value_by_isclose = result_compare_method.judge_value_by_isclose
    assert (
        judge_value_by_isclose(
            np.array([real], dtype=np.float32),
            np.array([1.0], dtype=np.float32),
            force_bf16=force_bf16,
        )
        == expected_pass
    )
