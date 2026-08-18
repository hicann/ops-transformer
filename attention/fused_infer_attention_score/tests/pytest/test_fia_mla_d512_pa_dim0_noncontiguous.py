#!/usr/bin/python3
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
"""arch22 D512 MLA PageAttention cache dim0-stride NPU regression tests.

The positive tests compare the same logical cache through three independent
oracles: contiguous NPU execution, dim0-strided NPU execution, and an FP32 CPU
PageAttention golden.  Padding between physical blocks is filled with poison
and checked after execution so an incorrect dense address calculation cannot
silently pass because the logical tensors happen to have the same shape.
"""

from dataclasses import dataclass
from itertools import accumulate
import math

import pytest

torch = pytest.importorskip("torch")
torch_npu = pytest.importorskip("torch_npu")


QK_DIM = 512
VALUE_DIM = 512
ROPE_DIM = 64
NUM_KV_HEADS = 1
NUM_QUERY_HEADS = 8
SCALE = 1.0 / math.sqrt(QK_DIM)


@dataclass(frozen=True)
class Geometry:
    name: str
    dtype: object
    query_layout: str
    cache_layout: str
    query_lengths: tuple
    kv_lengths: tuple
    block_size: int
    spare_blocks: int = 2
    table_order: str = "shuffled"


@dataclass
class Inputs:
    query: object
    key: object
    value: object
    query_rope: object
    key_rope: object
    block_table: object
    actual_query_lengths: object
    actual_kv_lengths: object


@dataclass
class Dim0View:
    view: object
    backing: object
    logical_block_elements: int
    stride0: int
    poison: float


STRIDE_PROFILES = (
    ("key-only", 17, 0, 0),
    ("value-only", 0, 29, 0),
    ("key-rope-only", 0, 0, 43),
    ("key-value-same", 23, 23, 0),
    ("key-greater-than-value", 61, 13, 0),
    ("value-greater-than-key", 13, 61, 0),
    ("key-and-key-rope", 19, 0, 47),
    ("value-and-key-rope", 0, 31, 53),
    ("all-different", 17, 37, 71),
)


def _require_arch22():
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        pytest.skip("Ascend NPU is unavailable")
    torch_npu.npu.set_device(0)
    device_name = torch_npu.npu.get_device_name(0).upper()
    if "910B" not in device_name:
        pytest.skip(f"arch22/Ascend 910B required, got {device_name}")
    if not hasattr(torch_npu, "npu_fused_infer_attention_score_v2"):
        pytest.skip("torch_npu lacks npu_fused_infer_attention_score_v2")


@pytest.fixture(scope="module", autouse=True)
def arch22_device():
    _require_arch22()
    yield
    torch.npu.synchronize()


def _format_cache(canonical, layout):
    # Canonical layout is [physicalBlock, blockSize, N, D].
    if layout == "BnBsH":
        return canonical.reshape(
            canonical.shape[0], canonical.shape[1], -1
        ).contiguous()
    if layout == "BnNBsD":
        return canonical.permute(0, 2, 1, 3).contiguous()
    if layout == "NZ":
        block_num, block_size, num_heads, head_dim = canonical.shape
        assert head_dim % 16 == 0
        return (
            canonical.reshape(block_num, block_size, num_heads, head_dim // 16, 16)
            .permute(0, 2, 3, 1, 4)
            .contiguous()
        )
    raise ValueError(f"unsupported cache layout {layout}")


def _canonical_cache(cache, layout):
    if layout == "BnBsH":
        return cache.reshape(cache.shape[0], cache.shape[1], 1, -1)
    if layout == "BnNBsD":
        return cache.permute(0, 2, 1, 3)
    if layout == "NZ":
        return cache.permute(0, 3, 1, 2, 4).reshape(
            cache.shape[0], cache.shape[3], cache.shape[1], -1
        )
    raise ValueError(f"unsupported cache layout {layout}")


def _build_block_table(geometry, generator):
    block_counts = [
        math.ceil(length / geometry.block_size) for length in geometry.kv_lengths
    ]
    required_blocks = sum(block_counts)
    physical_blocks = required_blocks + geometry.spare_blocks
    ids = torch.arange(physical_blocks, dtype=torch.int64)
    if geometry.table_order == "shuffled":
        ids = ids[torch.randperm(physical_blocks, generator=generator)]
    elif geometry.table_order == "reversed":
        ids = ids.flip(0)
    elif geometry.table_order != "ordered":
        raise ValueError(f"unsupported table order {geometry.table_order}")

    # Keep one unused table column as a -1 guard in addition to spare blocks.
    table = torch.full(
        (len(block_counts), max(block_counts) + 1), -1, dtype=torch.int32
    )
    cursor = 0
    for batch_index, count in enumerate(block_counts):
        table[batch_index, :count] = ids[cursor : cursor + count].to(torch.int32)
        cursor += count

    valid = table[table >= 0].to(torch.int64)
    assert valid.numel() == required_blocks
    assert valid.unique().numel() == valid.numel()
    assert bool(torch.any(valid > 0))
    return table, physical_blocks


def _build_inputs(geometry):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(20260813)
    block_table, physical_blocks = _build_block_table(geometry, generator)
    batch = len(geometry.query_lengths)
    max_query_length = max(geometry.query_lengths)

    query_parts = [
        torch.randn(
            length,
            NUM_QUERY_HEADS,
            QK_DIM,
            dtype=torch.float32,
            generator=generator,
        ).to(geometry.dtype)
        for length in geometry.query_lengths
    ]
    query_rope_parts = [
        torch.randn(
            length,
            NUM_QUERY_HEADS,
            ROPE_DIM,
            dtype=torch.float32,
            generator=generator,
        ).to(geometry.dtype)
        for length in geometry.query_lengths
    ]
    if geometry.query_layout == "TND":
        query = torch.cat(query_parts, dim=0)
        query_rope = torch.cat(query_rope_parts, dim=0)
        actual_query_lengths = torch.tensor(
            tuple(accumulate(geometry.query_lengths)), dtype=torch.int64
        )
    elif geometry.query_layout == "BNSD":
        assert len(set(geometry.query_lengths)) == 1
        query = torch.stack(query_parts, dim=0).permute(0, 2, 1, 3).contiguous()
        query_rope = (
            torch.stack(query_rope_parts, dim=0).permute(0, 2, 1, 3).contiguous()
        )
        actual_query_lengths = torch.tensor(geometry.query_lengths, dtype=torch.int64)
        assert query.shape[2] == max_query_length
    else:
        raise ValueError(f"unsupported query layout {geometry.query_layout}")

    def random_cache(head_dim):
        canonical = torch.randn(
            physical_blocks,
            geometry.block_size,
            NUM_KV_HEADS,
            head_dim,
            dtype=torch.float32,
            generator=generator,
        ).to(geometry.dtype)
        return _format_cache(canonical, geometry.cache_layout)

    return Inputs(
        query=query,
        key=random_cache(QK_DIM),
        value=random_cache(VALUE_DIM),
        query_rope=query_rope,
        key_rope=random_cache(ROPE_DIM),
        block_table=block_table,
        actual_query_lengths=actual_query_lengths,
        actual_kv_lengths=torch.tensor(geometry.kv_lengths, dtype=torch.int64),
    )


def _to_npu(inputs):
    return Inputs(
        query=inputs.query.npu(),
        key=inputs.key.npu(),
        value=inputs.value.npu(),
        query_rope=inputs.query_rope.npu(),
        key_rope=inputs.key_rope.npu(),
        block_table=inputs.block_table.npu(),
        # Sequence length metadata is a host input for this ACLNN API.
        actual_query_lengths=inputs.actual_query_lengths,
        actual_kv_lengths=inputs.actual_kv_lengths,
    )


def _make_dim0_view(source, gap_elements, poison):
    assert source.is_contiguous()
    assert source.shape[0] > 1
    block_num = source.shape[0]
    logical_block_elements = source[0].numel()
    stride0 = logical_block_elements + gap_elements
    storage_elements = (block_num - 1) * stride0 + logical_block_elements
    backing = torch.full(
        (storage_elements,), poison, dtype=source.dtype, device=source.device
    )
    view = torch.as_strided(
        backing,
        size=tuple(source.shape),
        stride=(stride0, *source.stride()[1:]),
    )
    view.copy_(source)
    assert not view.is_contiguous()
    assert view.stride(0) != source.stride(0)
    assert tuple(view.stride()[1:]) == tuple(source.stride()[1:])
    return Dim0View(
        view=view,
        backing=backing,
        logical_block_elements=logical_block_elements,
        stride0=stride0,
        poison=poison,
    )


def _assert_poison_untouched(cache_view):
    block_num = cache_view.view.shape[0]
    for block_index in range(block_num - 1):
        gap_start = block_index * cache_view.stride0 + cache_view.logical_block_elements
        gap_end = (block_index + 1) * cache_view.stride0
        gap = cache_view.backing[gap_start:gap_end]
        expected = torch.full_like(gap, cache_view.poison)
        assert torch.equal(gap, expected), (
            f"padding after physical block {block_index} was modified"
        )


def _cpu_golden(geometry, inputs):
    key = _canonical_cache(inputs.key, geometry.cache_layout)
    value = _canonical_cache(inputs.value, geometry.cache_layout)
    key_rope = _canonical_cache(inputs.key_rope, geometry.cache_layout)
    outputs = []

    if geometry.query_layout == "TND":
        query_parts = torch.split(inputs.query, geometry.query_lengths, dim=0)
        query_rope_parts = torch.split(inputs.query_rope, geometry.query_lengths, dim=0)
    else:
        query_parts = tuple(
            inputs.query[b].permute(1, 0, 2) for b in range(inputs.query.shape[0])
        )
        query_rope_parts = tuple(
            inputs.query_rope[b].permute(1, 0, 2)
            for b in range(inputs.query_rope.shape[0])
        )

    for batch_index, kv_length in enumerate(geometry.kv_lengths):
        block_count = math.ceil(kv_length / geometry.block_size)
        block_ids = inputs.block_table[batch_index, :block_count].long()
        batch_key = key.index_select(0, block_ids).reshape(-1, NUM_KV_HEADS, QK_DIM)[
            :kv_length
        ]
        batch_value = value.index_select(0, block_ids).reshape(
            -1, NUM_KV_HEADS, VALUE_DIM
        )[:kv_length]
        batch_key_rope = key_rope.index_select(0, block_ids).reshape(
            -1, NUM_KV_HEADS, ROPE_DIM
        )[:kv_length]

        query_total = (
            torch.cat((query_parts[batch_index], query_rope_parts[batch_index]), dim=-1)
            .float()
            .transpose(0, 1)
        )
        key_total = torch.cat((batch_key, batch_key_rope), dim=-1).float()
        key_total = key_total.permute(1, 0, 2).repeat_interleave(NUM_QUERY_HEADS, dim=0)
        value_by_head = (
            batch_value.float()
            .permute(1, 0, 2)
            .repeat_interleave(NUM_QUERY_HEADS, dim=0)
        )
        scores = torch.matmul(query_total, key_total.transpose(-2, -1))
        probabilities = torch.softmax(scores * SCALE, dim=-1)
        outputs.append(torch.matmul(probabilities, value_by_head).transpose(0, 1))

    if geometry.query_layout == "TND":
        return torch.cat(outputs, dim=0).to(geometry.dtype)
    return torch.stack(outputs, dim=0).permute(0, 2, 1, 3).to(geometry.dtype)


def _run(geometry, inputs, key=None, value=None, key_rope=None):
    torch.npu.synchronize()
    output, _ = torch_npu.npu_fused_infer_attention_score_v2(
        inputs.query,
        inputs.key if key is None else key,
        inputs.value if value is None else value,
        query_rope=inputs.query_rope,
        key_rope=inputs.key_rope if key_rope is None else key_rope,
        actual_seq_qlen=inputs.actual_query_lengths,
        actual_seq_kvlen=inputs.actual_kv_lengths,
        num_query_heads=NUM_QUERY_HEADS,
        num_key_value_heads=NUM_KV_HEADS,
        softmax_scale=SCALE,
        input_layout=geometry.query_layout,
        pre_tokens=65535,
        next_tokens=65535,
        block_table=inputs.block_table,
        block_size=geometry.block_size,
        return_softmax_lse=False,
    )
    torch.npu.synchronize()
    return output


def _golden_tolerance(dtype):
    if dtype == torch.float16:
        return 1e-3, 3e-2
    return 5e-3, 5e-2


def _run_positive(geometry, gaps):
    cpu_inputs = _build_inputs(geometry)
    npu_inputs = _to_npu(cpu_inputs)
    baseline = _run(geometry, npu_inputs)
    golden = _cpu_golden(geometry, cpu_inputs)
    rtol, atol = _golden_tolerance(geometry.dtype)
    torch.testing.assert_close(
        baseline.cpu(), golden, rtol=rtol, atol=atol, check_dtype=False
    )

    caches = (npu_inputs.key, npu_inputs.value, npu_inputs.key_rope)
    poisons = (37.0, -29.0, 53.0)
    views = [
        None if gap == 0 else _make_dim0_view(cache, gap, poison)
        for cache, gap, poison in zip(caches, gaps, poisons)
    ]
    test_caches = [
        cache if cache_view is None else cache_view.view
        for cache, cache_view in zip(caches, views)
    ]
    result = _run(geometry, npu_inputs, *test_caches)

    # The stride path must be numerically identical to the same NPU kernel with
    # dense caches; CPU golden remains an independent precision check.
    torch.testing.assert_close(
        result.cpu(), baseline.cpu(), rtol=0.0, atol=0.0, check_dtype=False
    )
    torch.testing.assert_close(
        result.cpu(), golden, rtol=rtol, atol=atol, check_dtype=False
    )
    for cache_view, source in zip(views, caches):
        if cache_view is not None:
            torch.testing.assert_close(
                cache_view.view.cpu(), source.cpu(), rtol=0.0, atol=0.0
            )
            _assert_poison_untouched(cache_view)


def test_fp16_tnd_bnbs_h_independent_stride_profiles():
    geometry = Geometry(
        name="fp16-tnd-bnbsh",
        dtype=torch.float16,
        query_layout="TND",
        cache_layout="BnBsH",
        query_lengths=(1, 2),
        kv_lengths=(127, 129),
        block_size=128,
        spare_blocks=3,
    )
    # This loop shares the geometry but deliberately rebuilds randomized input
    # for isolation: every profile independently checks its continuous oracle.
    for _, key_gap, value_gap, key_rope_gap in STRIDE_PROFILES:
        _run_positive(geometry, (key_gap, value_gap, key_rope_gap))


@pytest.mark.parametrize(
    "geometry",
    (
        Geometry(
            "fp16-single-block-exact-boundary",
            torch.float16,
            "TND",
            "BnBsH",
            (1,),
            (128,),
            128,
        ),
        Geometry(
            "bf16-bnsd-bnnbds-multi-batch",
            torch.bfloat16,
            "BNSD",
            "BnNBsD",
            (2, 2),
            (255, 385),
            128,
        ),
        Geometry(
            "bf16-bnsd-nz-tail-block",
            torch.bfloat16,
            "BNSD",
            "NZ",
            (2, 2),
            (17, 129),
            128,
        ),
        Geometry(
            "fp16-long-kv-flash-decode",
            torch.float16,
            "TND",
            "BnNBsD",
            (1,),
            (4097,),
            128,
            table_order="reversed",
        ),
    ),
    ids=lambda geometry: geometry.name,
)
def test_arch22_d512_mla_geometry_and_layout_coverage(geometry):
    _run_positive(geometry, (17, 37, 71))


def _make_inner_strided(source):
    padded_shape = list(source.shape)
    padded_shape[-1] += 1
    padded = torch.zeros(padded_shape, dtype=source.dtype, device=source.device)
    slices = [slice(None)] * source.ndim
    slices[-1] = slice(0, source.shape[-1])
    view = padded[tuple(slices)]
    view.copy_(source)
    assert not view.is_contiguous()
    assert tuple(view.shape) == tuple(source.shape)
    return view


@pytest.mark.parametrize(
    "targets",
    (("key",), ("value",), ("key_rope",), ("key", "value", "key_rope")),
    ids=("key", "value", "key-rope", "all-caches"),
)
def test_inner_axis_noncontiguous_cache_falls_back_to_contiguous(targets):
    geometry = Geometry(
        "inner-axis-contiguous-fallback",
        torch.float16,
        "TND",
        "BnNBsD",
        (1,),
        (129,),
        128,
    )
    npu_inputs = _to_npu(_build_inputs(geometry))
    baseline = _run(geometry, npu_inputs)
    caches = {
        "key": npu_inputs.key,
        "value": npu_inputs.value,
        "key_rope": npu_inputs.key_rope,
    }
    for target in targets:
        caches[target] = _make_inner_strided(caches[target])
    result = _run(
        geometry,
        npu_inputs,
        caches["key"],
        caches["value"],
        caches["key_rope"],
    )
    torch.testing.assert_close(
        result.cpu(), baseline.cpu(), rtol=0.0, atol=0.0, check_dtype=False
    )
