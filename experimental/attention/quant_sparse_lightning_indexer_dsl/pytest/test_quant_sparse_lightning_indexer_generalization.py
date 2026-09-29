# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

import copy
import gc
import sys
from itertools import product
from pathlib import Path

import pytest
from qsli_test_utils import generate_qsli_test_data, run_qsli_case, compare_qsli_case
from quant_sparse_lightning_indexer_golden import pack_candidate_k
import torch
import torch_npu


TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIR))

from qsli_test_utils import result_compare_method
from quant_sparse_lightning_indexer_golden import (  # noqa: E402
    quant_sparse_lightning_indexer_golden,
    quant_sparse_lightning_indexer_golden_selected,
)
import cann_ops_transformer.ops.attention.quant_sparse_lightning_indexer_dsl  # noqa: F401,E402


quant_sparse_lightning_indexer = (
    torch.ops.cann_ops_transformer.ds41.quant_sparse_lightning_indexer
)


BATCH_SIZES = (1, 2, 4, 8, 16, 32)
SEQUENCE_LENGTHS = (1024, 2048, 4096, 8192, 16384, 32768, 131072)
CMP_RATIOS = (1, 2)
DECODE_CASES = tuple(product(BATCH_SIZES, SEQUENCE_LENGTHS, CMP_RATIOS))
PREFILL_CASES = tuple(product(BATCH_SIZES, SEQUENCE_LENGTHS, CMP_RATIOS))

PAGE_SIZE = 128
MASK_TAIL_S2 = (
    1,
    5,
    7,
    8,
    9,
    127,
    128,
    129,
    511,
    512,
    513,
    2047,
    2048,
    2049,
)
MASK_TAIL_STORAGE_CASES = tuple(("contiguous", s2) for s2 in MASK_TAIL_S2) + tuple(
    ("dim0_noncontiguous", s2) for s2 in MASK_TAIL_S2 if s2 > PAGE_SIZE
)
PACKED_D = 64
N1 = 32
N2 = 1
TOPK = 512
CANDIDATE_CAPACITY = 2048
CANDIDATE_BLOCK_SIZE = 8
PREFILL_QUERY_LENGTH = 8192
STORAGE_MODES = ("contiguous", "dim0_noncontiguous")
REQUESTED_SHAPE_CASES = (
    (1, 4, 2048),
    (4, 4, 4096),
    (8, 4, 8192),
    (16, 4, 16384),
    (32, 4, 32768),
    (1, 8192, 2048),
    (4, 8192, 4096),
    (8, 8192, 8192),
    (16, 8192, 16384),
    (32, 8192, 32768),
)


def _case_id(case: tuple[int, int, int]) -> str:
    batch, s2, cmp_ratio = case
    return f"B{batch}-S2_{s2 // 1024}K-R{cmp_ratio}"


def _make_case(
    batch: int,
    s2: int,
    cmp_ratio: int,
    query_length: int,
    storage_mode: str = "contiguous",
) -> dict[str, torch.Tensor]:
    """Create packed MXFP4 data without an intermediate FP32 K tensor."""
    seed = batch * 1_000_000 + query_length * 131 + s2 + cmp_ratio
    generator = torch.Generator(device="cpu").manual_seed(seed)
    total_query = batch * query_length
    pages_per_batch = (s2 + PAGE_SIZE - 1) // PAGE_SIZE
    logical_candidate_blocks = (s2 + CANDIDATE_BLOCK_SIZE - 1) // CANDIDATE_BLOCK_SIZE
    valid_candidates = min(CANDIDATE_CAPACITY, logical_candidate_blocks)

    # Physical PA pages are intentionally shared by batches. Block-table rows
    # remain independent logical mappings, while peak memory depends on S2
    # instead of B*S2.
    q = torch.randint(
        0,
        256,
        (total_query, N1, PACKED_D),
        generator=generator,
        dtype=torch.uint8,
    )
    k = torch.randint(
        0,
        256,
        (pages_per_batch, PAGE_SIZE, N2, PACKED_D),
        generator=generator,
        dtype=torch.uint8,
    )
    weights = torch.rand((total_query, N1), generator=generator, dtype=torch.float32)
    descale_q = torch.randint(
        125,
        130,
        (total_query, N1, 2, 2),
        generator=generator,
        dtype=torch.uint8,
    )
    descale_k = torch.randint(
        125,
        130,
        (pages_per_batch, PAGE_SIZE, N2, 2, 2),
        generator=generator,
        dtype=torch.uint8,
    )
    block_fusion = None
    key_storage = None
    scale_storage = None
    if storage_mode == "dim0_noncontiguous":
        key_elements = PAGE_SIZE * N2 * PACKED_D
        scale_elements = PAGE_SIZE * N2 * 2 * 2
        block_fusion = torch.empty(
            (pages_per_batch, key_elements + scale_elements),
            dtype=torch.uint8,
        )
        block_fusion[:, :key_elements] = k.reshape(pages_per_batch, -1)
        block_fusion[:, key_elements:] = descale_k.reshape(pages_per_batch, -1)
        k = block_fusion[:, :key_elements].view(
            pages_per_batch, PAGE_SIZE, N2, PACKED_D
        )
        descale_k = block_fusion[:, key_elements:].view(
            pages_per_batch, PAGE_SIZE, N2, 2, 2
        )
    elif storage_mode == "key_dim0_noncontiguous":
        key_elements = PAGE_SIZE * N2 * PACKED_D
        key_storage = torch.empty(
            (pages_per_batch, key_elements + 64), dtype=torch.uint8
        )
        key_storage[:, :key_elements] = k.reshape(pages_per_batch, -1)
        k = key_storage[:, :key_elements].view(pages_per_batch, PAGE_SIZE, N2, PACKED_D)
    elif storage_mode == "scale_dim0_noncontiguous":
        scale_elements = PAGE_SIZE * N2 * 2 * 2
        scale_storage = torch.empty(
            (pages_per_batch, scale_elements + 32), dtype=torch.uint8
        )
        scale_storage[:, :scale_elements] = descale_k.reshape(pages_per_batch, -1)
        descale_k = scale_storage[:, :scale_elements].view(
            pages_per_batch, PAGE_SIZE, N2, 2, 2
        )
    elif storage_mode != "contiguous":
        raise ValueError(f"unsupported storage_mode: {storage_mode}")
    block_table = torch.stack(
        [
            torch.randperm(pages_per_batch, generator=generator).to(torch.int32)
            for _ in range(batch)
        ]
    )
    candidate_per_batch = torch.stack(
        [
            torch.randperm(logical_candidate_blocks, generator=generator)[
                :valid_candidates
            ].to(torch.int32)
            for _ in range(batch)
        ]
    )
    candidate = torch.full((total_query, N2, CANDIDATE_CAPACITY), -1, dtype=torch.int32)
    for batch_index in range(batch):
        begin = batch_index * query_length
        candidate[begin : begin + query_length, 0, :valid_candidates] = (
            candidate_per_batch[batch_index]
        )
    cu = torch.arange(batch + 1, dtype=torch.int32) * query_length
    return {
        "q": q,
        "k": k,
        "weights": weights,
        "descale_q": descale_q,
        "descale_k": descale_k,
        "candidate": candidate,
        "candidate_length": torch.full(
            (total_query, N2), valid_candidates, dtype=torch.int32
        ),
        "cu": cu,
        "used_q": torch.full((batch,), query_length, dtype=torch.int32),
        "used_k": torch.full((batch,), s2, dtype=torch.int32),
        "residual": (
            None
            if cmp_ratio == 1
            else torch.arange(batch, dtype=torch.int32) % cmp_ratio
        ),
        "block_table": block_table,
        "offset": (
            torch.arange(total_query, dtype=torch.int32).reshape(-1, 1) * 3 + 37
        ),
        "_block_fusion": block_fusion,
        "_key_storage": key_storage,
        "_scale_storage": scale_storage,
    }


def _npu_arguments(case: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    block_fusion = case.get("_block_fusion")
    key_storage = case.get("_key_storage")
    scale_storage = case.get("_scale_storage")
    npu = {
        name: value.npu() if isinstance(value, torch.Tensor) else value
        for name, value in case.items()
        if name
        not in (
            "k",
            "descale_k",
            "_block_fusion",
            "_key_storage",
            "_scale_storage",
        )
    }
    pages = int(case["k"].shape[0])
    key_elements = PAGE_SIZE * N2 * PACKED_D
    scale_elements = PAGE_SIZE * N2 * 2 * 2
    if block_fusion is not None:
        # Match ASC: move the fused physical allocation once, then create the
        # strided K/KScale views on NPU so .npu() cannot compact either view.
        block_fusion_npu = block_fusion.npu()
        npu["k"] = block_fusion_npu[:, :key_elements].view(
            pages, PAGE_SIZE, N2, PACKED_D
        )
        npu["descale_k"] = block_fusion_npu[:, key_elements:].view(
            pages, PAGE_SIZE, N2, 2, 2
        )
        npu["_block_fusion"] = block_fusion_npu
    else:
        if key_storage is None:
            npu["k"] = case["k"].npu()
        else:
            key_storage_npu = key_storage.npu()
            npu["k"] = key_storage_npu[:, :key_elements].view(
                pages, PAGE_SIZE, N2, PACKED_D
            )
            npu["_key_storage"] = key_storage_npu
        if scale_storage is None:
            npu["descale_k"] = case["descale_k"].npu()
        else:
            scale_storage_npu = scale_storage.npu()
            npu["descale_k"] = scale_storage_npu[:, :scale_elements].view(
                pages, PAGE_SIZE, N2, 2, 2
            )
            npu["_scale_storage"] = scale_storage_npu
    if block_fusion is not None or key_storage is not None:
        assert npu["k"].stride(0) > key_elements
    if block_fusion is not None or scale_storage is not None:
        assert npu["descale_k"].stride(0) > scale_elements
    return npu


def _remove_offset(indices: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    logical = indices.clone()
    expanded = offsets.reshape(indices.shape[0], 1, 1).expand_as(indices)
    valid = logical >= 0
    logical[valid] -= expanded[valid]
    return logical


def _result_compare_params(rows: int, s2: int, cmp_ratio: int) -> tuple:
    return (
        rows,
        1,
        s2,
        rows,
        rows * s2,
        N1,
        N2,
        128,
        PAGE_SIZE,
        (s2 + PAGE_SIZE - 1) // PAGE_SIZE,
        torch.uint8,
        torch.float32,
        torch.uint8,
        torch.int32,
        list(range(1, rows + 1)),
        [s2] * rows,
        1,
        1,
        "TND",
        "PA_BBND",
        TOPK,
        3,
        (-1, 1),
        (-1, 1),
        (0, 1),
        (125, 129),
        (125, 129),
    )


def _compare(
    golden: dict[str, torch.Tensor],
    actual_indices: torch.Tensor,
    actual_values: torch.Tensor,
    s2: int,
    cmp_ratio: int,
) -> None:
    expected_indices = _remove_offset(
        golden["sparse_indices"], golden["_output_idx_offset"]
    )
    logical_actual = _remove_offset(actual_indices, golden["_output_idx_offset"])
    result, fulfill_percent = result_compare_method.check_result(
        expected_indices,
        logical_actual,
        golden["_topk_value"],
        _result_compare_params(int(actual_indices.shape[0]), s2, cmp_ratio),
    )
    assert result == "Pass", f"TopK compare failed: {fulfill_percent:.6f}%"

    reference_values = torch.zeros_like(actual_values, dtype=torch.float32)
    for row in range(actual_indices.shape[0]):
        valid = logical_actual[row, 0] >= 0
        reference_values[row, 0, valid] = golden["_topk_value"][
            row, 0, 0, logical_actual[row, 0, valid].long()
        ]
    assert result_compare_method.judge_value_by_isclose(
        actual_values.float().numpy().reshape(-1),
        reference_values.numpy().reshape(-1),
    ), "sparse_values failed result_compare_method"


def _assert_invalid_tail_semantics(
    expected_indices: torch.Tensor,
    expected_values: torch.Tensor,
    actual_indices: torch.Tensor,
    actual_values: torch.Tensor,
) -> None:
    """Check ASC-compatible valid-prefix plus -1/zero-tail semantics."""
    expected_valid = (expected_indices >= 0).sum(dim=-1)
    actual_valid = (actual_indices >= 0).sum(dim=-1)
    assert torch.equal(actual_valid, expected_valid), (
        "NPU valid TopK count differs from golden"
    )

    positions = torch.arange(TOPK).reshape(1, 1, TOPK)
    valid_prefix = positions < expected_valid.unsqueeze(-1)
    assert torch.equal(expected_indices >= 0, valid_prefix), (
        "golden valid entries must be a prefix followed by -1"
    )
    assert torch.equal(actual_indices >= 0, valid_prefix), (
        "NPU valid entries must be a prefix followed by -1"
    )
    assert torch.all(expected_indices[~valid_prefix] == -1)
    assert torch.all(actual_indices[~valid_prefix] == -1)
    assert torch.all(expected_values[~valid_prefix] == 0)
    assert torch.all(actual_values[~valid_prefix] == 0)


def _selected_query_offsets(query_length: int, s2: int) -> list[int]:
    """Select causal rows covering empty, partial, and full TopK states."""
    if query_length <= 4:
        return list(range(query_length))

    selected = []
    visible_at_first = max(0, s2 - query_length + 1)
    if visible_at_first == 0:
        selected.append(0)

    if visible_at_first < TOPK:
        partial_visible = max(1, min(TOPK - 1, TOPK // 2))
        partial_row = query_length - s2 + partial_visible - 1
        if 0 <= partial_row < query_length:
            selected.append(partial_row)

    first_full_row = max(0, query_length - s2 + TOPK - 1)
    if first_full_row < query_length:
        selected.append(first_full_row)
    selected.extend((query_length // 2, query_length - 1))
    return list(dict.fromkeys(selected))


def _flatten_full_golden(golden: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Convert a batched full golden into selected-row result_compare layout."""
    cu = golden["_cu_seqlens_q"].tolist()
    full_score = golden["_topk_value"]
    rows = int(golden["sparse_indices"].shape[0])
    flat_score = torch.full(
        (rows, 1, 1, int(full_score.shape[-1])),
        -float("inf"),
        dtype=full_score.dtype,
    )
    for batch, (begin, end) in enumerate(zip(cu, cu[1:])):
        flat_score[begin:end, 0, 0] = full_score[batch, 0, : end - begin]
    return {**golden, "_topk_value": flat_score}


def _invoke_operator(npu: dict[str, torch.Tensor], query_length: int, cmp_ratio: int):
    metadata = (
        torch.ops.cann_ops_transformer.ds41.quant_sparse_lightning_indexer_metadata(
            npu["candidate_length"],
            cu_seqlens_q=npu["cu"],
            seqused_q=npu["used_q"],
            seqused_k=npu["used_k"],
            cmp_residual_k=npu["residual"],
            max_seqlen_q=query_length,
            max_seqlen_k=npu["block_table"].shape[1] * npu["k"].shape[1],
            num_heads_q=N1,
            num_heads_k=N2,
            head_dim=PACKED_D * 2,
            topk=TOPK,
            mask_mode=3,
            cmp_ratio=cmp_ratio,
            quant_mode=1,
            candidate_block_size=CANDIDATE_BLOCK_SIZE,
            layout_k="PA_BBND",
        )
    )
    return quant_sparse_lightning_indexer(
        npu["q"],
        pack_candidate_k(npu["k"], npu["descale_k"]),
        npu["weights"],
        npu["descale_q"],
        npu["candidate"],
        npu["candidate_length"],
        TOPK,
        1,
        CANDIDATE_BLOCK_SIZE,
        cu_seqlens_q=npu["cu"],
        seqused_q=npu["used_q"],
        seqused_k=npu["used_k"],
        cmp_residual_k=npu["residual"],
        block_table=npu["block_table"],
        output_idx_offset=npu["offset"],
        metadata=metadata,
        layout_k="PA_BBND",
        max_seqlen_q=query_length,
        mask_mode=3,
        cmp_ratio=cmp_ratio,
        return_value=True,
    )


def _run_operator(
    case: dict[str, torch.Tensor], query_length: int, cmp_ratio: int
) -> dict[str, torch.Tensor]:
    actual_outputs = _invoke_operator(_npu_arguments(case), query_length, cmp_ratio)
    torch.npu.synchronize()
    return {
        name: value.cpu()
        for name, value in zip(("sparse_indices", "sparse_values"), actual_outputs)
    }


@pytest.fixture(autouse=True)
def _release_case_memory():
    yield
    gc.collect()
    torch.npu.empty_cache()


@pytest.mark.npu
@pytest.mark.parametrize(
    "batch,s2,cmp_ratio",
    DECODE_CASES,
    ids=[_case_id(case) for case in DECODE_CASES],
)
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_qsli_decode_cross_precision(
    batch: int, s2: int, cmp_ratio: int, storage_mode: str
):
    case = _make_case(batch, s2, cmp_ratio, query_length=1, storage_mode=storage_mode)
    golden = quant_sparse_lightning_indexer_golden(
        case["q"],
        pack_candidate_k(case["k"], case["descale_k"]),
        case["weights"],
        case["descale_q"],
        case["candidate"],
        case["candidate_length"],
        cu_seqlens_q=case["cu"],
        seqused_q=case["used_q"],
        seqused_k=case["used_k"],
        cmp_residual_k=case["residual"],
        block_table=case["block_table"],
        output_idx_offset=case["offset"],
        topk=TOPK,
        candidate_block_size=CANDIDATE_BLOCK_SIZE,
        max_seqlen_q=1,
        mask_mode=3,
        cmp_ratio=cmp_ratio,
        return_value=True,
    )
    actual = _run_operator(case, query_length=1, cmp_ratio=cmp_ratio)
    _compare(
        golden,
        actual["sparse_indices"],
        actual["sparse_values"],
        s2,
        cmp_ratio,
    )


@pytest.mark.npu
@pytest.mark.slow
@pytest.mark.parametrize(
    "batch,s2,cmp_ratio",
    PREFILL_CASES,
    ids=[_case_id(case) for case in PREFILL_CASES],
)
@pytest.mark.parametrize("storage_mode", STORAGE_MODES)
def test_qsli_prefill_cross_precision(
    batch: int, s2: int, cmp_ratio: int, storage_mode: str
):
    case = _make_case(
        batch,
        s2,
        cmp_ratio,
        PREFILL_QUERY_LENGTH,
        storage_mode=storage_mode,
    )
    selected_rows = []
    for batch_index in range(batch):
        begin = batch_index * PREFILL_QUERY_LENGTH
        selected_rows.extend(
            (begin, begin + PREFILL_QUERY_LENGTH // 2, begin + PREFILL_QUERY_LENGTH - 1)
        )
    golden = quant_sparse_lightning_indexer_golden_selected(
        case["q"],
        pack_candidate_k(case["k"], case["descale_k"]),
        case["weights"],
        case["descale_q"],
        case["candidate"],
        case["candidate_length"],
        selected_rows,
        cu_seqlens_q=case["cu"],
        seqused_q=case["used_q"],
        seqused_k=case["used_k"],
        cmp_residual_k=case["residual"],
        block_table=case["block_table"],
        output_idx_offset=case["offset"],
        topk=TOPK,
        candidate_block_size=CANDIDATE_BLOCK_SIZE,
        max_seqlen_q=PREFILL_QUERY_LENGTH,
        mask_mode=3,
        cmp_ratio=cmp_ratio,
        return_value=True,
    )
    actual = _run_operator(case, query_length=PREFILL_QUERY_LENGTH, cmp_ratio=cmp_ratio)
    selected = torch.tensor(selected_rows, dtype=torch.long)
    _compare(
        golden,
        actual["sparse_indices"].index_select(0, selected),
        actual["sparse_values"].index_select(0, selected),
        s2,
        cmp_ratio,
    )


@pytest.mark.npu
@pytest.mark.slow
@pytest.mark.parametrize(
    "batch,query_length,s2",
    REQUESTED_SHAPE_CASES,
    ids=[
        f"B{batch}-S1_{query_length}-S2_{s2 // 1024}K"
        for batch, query_length, s2 in REQUESTED_SHAPE_CASES
    ],
)
def test_qsli_requested_shape_precision(batch: int, query_length: int, s2: int):
    """Requested paired B/S1/S2 regression, using selected-row golden for 8K."""
    case = _make_case(
        batch,
        s2,
        cmp_ratio=1,
        query_length=query_length,
        storage_mode="contiguous",
    )
    if query_length == 4:
        selected_rows = list(range(batch * query_length))
    else:
        selected_rows = []
        for batch_index in range(batch):
            begin = batch_index * query_length
            selected_rows.extend(
                begin + offset for offset in _selected_query_offsets(query_length, s2)
            )
    golden = quant_sparse_lightning_indexer_golden_selected(
        case["q"],
        pack_candidate_k(case["k"], case["descale_k"]),
        case["weights"],
        case["descale_q"],
        case["candidate"],
        case["candidate_length"],
        selected_rows,
        cu_seqlens_q=case["cu"],
        seqused_q=case["used_q"],
        seqused_k=case["used_k"],
        cmp_residual_k=None,
        block_table=case["block_table"],
        output_idx_offset=case["offset"],
        topk=TOPK,
        candidate_block_size=CANDIDATE_BLOCK_SIZE,
        max_seqlen_q=query_length,
        mask_mode=3,
        cmp_ratio=1,
        return_value=True,
    )
    actual = _run_operator(case, query_length=query_length, cmp_ratio=1)
    selected = torch.tensor(selected_rows, dtype=torch.long)
    selected_actual_indices = actual["sparse_indices"].index_select(0, selected)
    selected_actual_values = actual["sparse_values"].index_select(0, selected)
    _assert_invalid_tail_semantics(
        golden["sparse_indices"],
        golden["sparse_values"],
        selected_actual_indices,
        selected_actual_values,
    )
    _compare(
        golden,
        selected_actual_indices,
        selected_actual_values,
        s2,
        1,
    )


@pytest.mark.npu
def test_qsli_invalid_query_and_minus_one_precision():
    """Align ASC invalid-row, padding-row, and insufficient-TopK behavior."""
    batch, query_length, s2 = 3, 4, 513
    case = _make_case(
        batch,
        s2,
        cmp_ratio=1,
        query_length=query_length,
        storage_mode="contiguous",
    )
    case["used_q"] = torch.tensor([0, 2, 4], dtype=torch.int32)
    candidate_empty_row = query_length + 1
    case["candidate_length"][candidate_empty_row, 0] = 0

    golden = quant_sparse_lightning_indexer_golden(
        case["q"],
        pack_candidate_k(case["k"], case["descale_k"]),
        case["weights"],
        case["descale_q"],
        case["candidate"],
        case["candidate_length"],
        cu_seqlens_q=case["cu"],
        seqused_q=case["used_q"],
        seqused_k=case["used_k"],
        cmp_residual_k=None,
        block_table=case["block_table"],
        output_idx_offset=case["offset"],
        topk=TOPK,
        candidate_block_size=CANDIDATE_BLOCK_SIZE,
        max_seqlen_q=query_length,
        mask_mode=3,
        cmp_ratio=1,
        return_value=True,
    )
    actual = _run_operator(case, query_length=query_length, cmp_ratio=1)
    _assert_invalid_tail_semantics(
        golden["sparse_indices"],
        golden["sparse_values"],
        actual["sparse_indices"],
        actual["sparse_values"],
    )
    _compare(
        _flatten_full_golden(golden),
        actual["sparse_indices"],
        actual["sparse_values"],
        s2,
        1,
    )


@pytest.mark.npu
@pytest.mark.parametrize(
    "storage_mode",
    ("key_dim0_noncontiguous", "scale_dim0_noncontiguous"),
)
def test_qsli_independent_dim0_stride_precision(storage_mode: str):
    case = _make_case(2, 1024, 2, query_length=1, storage_mode=storage_mode)
    golden = quant_sparse_lightning_indexer_golden(
        case["q"],
        pack_candidate_k(case["k"], case["descale_k"]),
        case["weights"],
        case["descale_q"],
        case["candidate"],
        case["candidate_length"],
        cu_seqlens_q=case["cu"],
        seqused_q=case["used_q"],
        seqused_k=case["used_k"],
        cmp_residual_k=case["residual"],
        block_table=case["block_table"],
        output_idx_offset=case["offset"],
        topk=TOPK,
        candidate_block_size=CANDIDATE_BLOCK_SIZE,
        max_seqlen_q=1,
        mask_mode=3,
        cmp_ratio=2,
        return_value=True,
    )
    actual = _run_operator(case, query_length=1, cmp_ratio=2)
    _compare(golden, actual["sparse_indices"], actual["sparse_values"], 1024, 2)


@pytest.mark.npu
@pytest.mark.parametrize(
    "storage_mode,s2",
    MASK_TAIL_STORAGE_CASES,
    ids=[f"{storage_mode}-s2_{s2}" for storage_mode, s2 in MASK_TAIL_STORAGE_CASES],
)
@pytest.mark.parametrize("cmp_ratio", CMP_RATIOS)
@pytest.mark.parametrize("mask_mode", (0, 3))
def test_qsli_partial_s2_tail_precision(
    s2: int, cmp_ratio: int, storage_mode: str, mask_mode: int
):
    """Mask modes must honor tails at candidate, PA, top-k, and chunk edges."""
    case = _make_case(1, s2, cmp_ratio, query_length=1, storage_mode=storage_mode)
    residual = case["residual"] if mask_mode == 3 and cmp_ratio != 1 else None
    golden = quant_sparse_lightning_indexer_golden(
        case["q"],
        pack_candidate_k(case["k"], case["descale_k"]),
        case["weights"],
        case["descale_q"],
        case["candidate"],
        case["candidate_length"],
        cu_seqlens_q=case["cu"],
        seqused_q=case["used_q"],
        seqused_k=case["used_k"],
        cmp_residual_k=residual,
        block_table=case["block_table"],
        output_idx_offset=case["offset"],
        topk=TOPK,
        candidate_block_size=CANDIDATE_BLOCK_SIZE,
        max_seqlen_q=1,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        return_value=True,
    )
    npu = _npu_arguments(case)
    actual_outputs = quant_sparse_lightning_indexer(
        npu["q"],
        pack_candidate_k(npu["k"], npu["descale_k"]),
        npu["weights"],
        npu["descale_q"],
        npu["candidate"],
        npu["candidate_length"],
        TOPK,
        1,
        CANDIDATE_BLOCK_SIZE,
        cu_seqlens_q=npu["cu"],
        seqused_q=npu["used_q"],
        seqused_k=npu["used_k"],
        cmp_residual_k=(npu["residual"] if mask_mode == 3 and cmp_ratio != 1 else None),
        block_table=npu["block_table"],
        output_idx_offset=npu["offset"],
        max_seqlen_q=1,
        mask_mode=mask_mode,
        cmp_ratio=cmp_ratio,
        return_value=True,
        layout_k="PA_BBND",
    )
    torch.npu.synchronize()
    actual = {
        name: value.cpu()
        for name, value in zip(("sparse_indices", "sparse_values"), actual_outputs)
    }
    _compare(
        golden,
        actual["sparse_indices"],
        actual["sparse_values"],
        s2,
        cmp_ratio,
    )

    logical_indices = _remove_offset(
        actual["sparse_indices"], golden["_output_idx_offset"]
    )
    valid_indices = logical_indices[logical_indices >= 0]
    expected_count = min(s2, TOPK)
    assert valid_indices.numel() == expected_count
    assert torch.unique(valid_indices).numel() == expected_count
    assert torch.all(valid_indices < s2)
    if s2 <= TOPK:
        assert torch.equal(
            torch.sort(valid_indices).values,
            torch.arange(s2, dtype=torch.int32),
        )


@pytest.mark.npu
@pytest.mark.parametrize(
    "layout,input_name", [("TND", "k"), ("TND", "descale_k"), ("PA_BBND", "k")]
)
def test_qsli_rejects_inner_axis_noncontiguous(layout: str, input_name: str):
    from qsli_test_utils import _to_npu_tensors

    case = generate_qsli_test_data(
        dict(
            batch_size=1,
            query_length=1,
            sequence_length=1024,
            layout_k=layout,
        )
    )
    npu = _to_npu_tensors(case["tensors"])
    tensor = npu[input_name]
    storage = torch.empty(
        (*tensor.shape[:-1], tensor.shape[-1] * 2),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    npu[input_name] = storage[..., ::2]
    with pytest.raises(
        ValueError, match="only supports non-contiguous storage on axis 0"
    ):
        quant_sparse_lightning_indexer(
            npu["q"],
            npu["k"],
            npu["weights"],
            npu["descale_q"],
            npu["candidate"],
            npu["candidate_length"],
            TOPK,
            1,
            CANDIDATE_BLOCK_SIZE,
            descale_k=npu.get("descale_k"),
            cu_seqlens_q=npu["cu"],
            cu_seqlens_k=npu["cu_k"],
            seqused_q=npu["used_q"],
            seqused_k=npu["used_k"],
            block_table=npu["block_table"],
            max_seqlen_q=1,
            layout_k=layout,
        )


@pytest.mark.npu
@pytest.mark.parametrize("page", [64, 128])
@pytest.mark.parametrize("mask", [0, 3])
@pytest.mark.parametrize("ratio", [1, 2])
@pytest.mark.parametrize("offset", [False, True])
def test_packed_k_static_combinations(page, mask, ratio, offset):
    case = generate_qsli_test_data(
        dict(
            batch_size=1,
            query_length=4,
            sequence_length=8191,
            block_size=page,
            block_num=(8191 + page - 1) // page,
            mask_mode=mask,
            cmp_ratio=ratio,
            topk=512,
            candidate_block_length=[[0], [1], [17], [1024]],
            cmp_residual_k=[0] if mask == 3 and ratio != 1 else None,
            output_idx_offset=offset,
            storage_mode="dim0_noncontiguous" if offset else "contiguous",
            return_value=True,
            seed=20260917,
        )
    )
    assert case["tensors"]["k"].shape[1:] == (page // 8, 544)
    assert "descale_k" not in case["tensors"]
    for return_value in (True, False):
        case["params"]["return_value"] = return_value
        for _ in range(2):
            actual = run_qsli_case(case)
            compare_qsli_case(case, actual)


@pytest.fixture(scope="module")
def batch_level():
    previous = torch_npu.npu._get_deterministic_level()
    try:
        try:
            torch_npu.npu.set_deterministic_level(3)
            torch.ones(1, device="npu")
            torch.npu.synchronize()
        except RuntimeError as error:
            if "deterministic" not in str(error).lower():
                raise
            pytest.skip("The installed runtime does not support deterministic level 3")
        yield
    finally:
        torch_npu.npu.set_deterministic_level(previous)


@pytest.mark.npu
@pytest.mark.parametrize("page", [64, 128])
def test_batch_consistency(page, batch_level):
    torch_npu.npu.set_deterministic_level(3)
    import ops.quant_sparse_lightning_indexer_dsl as impl

    case = generate_qsli_test_data(
        dict(
            batch_size=1,
            query_length=6,
            sequence_length=8192,
            block_size=page,
            block_num=8192 // page,
            mask_mode=3,
            cmp_ratio=2,
            topk=512,
            cmp_residual_k=[0],
            return_value=True,
            seed=20260918,
        )
    )
    reference = run_qsli_case(case)
    compare_qsli_case(case, reference)
    compiled = dict(impl._COMPILED_KERNEL)
    for batch in (1, 2, 12, 32):
        expanded = copy.deepcopy(case)
        expanded["params"].update(batch_size=batch, q_t_size=batch * 6)
        for name in (
            "q",
            "weights",
            "descale_q",
            "candidate",
            "candidate_length",
            "used_q",
            "used_k",
            "residual",
            "block_table",
            "offset",
        ):
            value = expanded["tensors"].get(name)
            if value is not None:
                expanded["tensors"][name] = value.repeat(
                    (batch,) + (1,) * (value.ndim - 1)
                )
        expanded["tensors"]["cu"] = torch.arange(batch + 1, dtype=torch.int32) * 6
        expanded["tensors"]["cu_k"] = torch.arange(batch + 1, dtype=torch.int32) * 8192
        actual = run_qsli_case(expanded)
        for name in ("sparse_indices", "sparse_values"):
            expected = reference[name].repeat(
                (batch,) + (1,) * (reference[name].ndim - 1)
            )
            assert torch.equal(
                actual[name].view(torch.uint8), expected.view(torch.uint8)
            ), (page, batch, name)
        assert compiled.keys() == impl._COMPILED_KERNEL.keys()
        assert all(
            impl._COMPILED_KERNEL[key] is value for key, value in compiled.items()
        )
    # Both runtime branches must reuse exactly the same compiled executable.
    torch_npu.npu.set_deterministic_level(0)
    ordinary = run_qsli_case(case)
    for name in reference:
        assert torch.equal(
            ordinary[name].view(torch.uint8), reference[name].view(torch.uint8)
        )
    assert compiled.keys() == impl._COMPILED_KERNEL.keys()
    assert all(impl._COMPILED_KERNEL[key] is value for key, value in compiled.items())
