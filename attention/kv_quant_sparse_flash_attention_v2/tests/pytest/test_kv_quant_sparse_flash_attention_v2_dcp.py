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
import torch_npu

from cann_ops_transformer.ops.attention.kv_quant_sparse_flash_attention_v2 import (
    kv_quant_sparse_flash_attention,
)


@pytest.fixture(autouse=True)
def _require_arch22_device():
    device_name = torch.npu.get_device_name(0)
    if "910B" not in device_name and "910_93" not in device_name:
        pytest.skip("INT8 arch22 DCP regression")


def _inputs(
    layout,
    heads,
    sparse_count,
    query_count,
    kv_dtype,
    query_dtype,
    has_sinks,
    valid_count=None,
    default_zero_sink=True,
):
    torch.manual_seed(20261030)
    batch_size = 2
    page_size = 128
    latent = torch.randint(-4, 5, (sparse_count, 512)).to(kv_dtype)
    rope = torch.randn(sparse_count, 64).to(query_dtype)
    scales = torch.rand(sparse_count, 4) + 0.5
    packed = torch.cat(
        [latent.view(torch.uint8), rope.view(torch.uint8), scales.view(torch.uint8)],
        dim=-1,
    ).view(kv_dtype)
    query_row = torch.randn(heads, 576).to(query_dtype)
    query = query_row.expand(batch_size, query_count, heads, 576).contiguous()
    indices = (
        torch.arange(sparse_count, dtype=torch.int32)
        .expand(batch_size, query_count, 1, sparse_count)
        .clone()
    )
    if layout == "PA_BSND":
        key = packed.reshape(-1, page_size, 1, 656)
        block_table = (
            torch.arange(key.shape[0], dtype=torch.int32)
            .expand(batch_size, -1)
            .contiguous()
        )
        query_layout = "TND"
        kv_lengths = [sparse_count] * batch_size
    else:
        key = (
            packed.reshape(1, sparse_count, 1, 656)
            .expand(batch_size, -1, -1, -1)
            .contiguous()
        )
        block_table = None
        query_layout = layout
        kv_lengths = [sparse_count] * batch_size
        if layout == "TND":
            key = key.flatten(0, 1)
            kv_lengths = [sparse_count, batch_size * sparse_count]
    query_lengths = [query_count] * batch_size
    if query_layout == "TND":
        query = query.flatten(0, 1)
        indices = indices.flatten(0, 1)
        query_lengths = [query_count, batch_size * query_count]
    sinks = torch.linspace(-1, 1, heads) if has_sinks else None
    kwargs = dict(
        query=query.npu(),
        key=key.npu(),
        sparse_indices=indices.npu(),
        scale_value=576**-0.5,
        key_quant_mode=2,
        value_quant_mode=2,
        block_table=block_table.npu() if block_table is not None else None,
        actual_seq_lengths_query=torch.tensor(query_lengths, dtype=torch.int32).npu(),
        actual_seq_lengths_kv=torch.tensor(kv_lengths, dtype=torch.int32).npu(),
        sparse_block_size=1,
        layout_query=query_layout,
        layout_kv=layout,
        sparse_mode=0,
        attention_mode=2,
        quant_scale_repo_mode=1,
        tile_size=128,
        rope_head_dim=64,
        sinks=sinks.npu() if sinks is not None else None,
        return_softmax_lse=True,
    )
    kwargs["value"] = kwargs["key"]
    dequantized = (
        (latent.float() * scales.repeat_interleave(128, dim=-1)).to(query_dtype).float()
    )
    reference_key = torch.cat([dequantized, rope.float()], dim=-1)
    if valid_count is not None:
        reference_key = reference_key[:valid_count]
        dequantized = dequantized[:valid_count]
    scores = query_row.float() @ reference_key.T * kwargs["scale_value"]
    if sinks is None and default_zero_sink:
        sinks = torch.zeros(heads, dtype=torch.float32)
    if sinks is not None:
        scores = torch.cat([scores, sinks[:, None]], dim=-1)
        dequantized = torch.cat([dequantized, torch.zeros(1, 512)], dim=0)
    expected_attention = torch.softmax(scores, dim=-1) @ dequantized
    expected_lse = torch.logsumexp(scores, dim=-1)
    return kwargs, expected_attention, expected_lse, sinks


@pytest.mark.ci
@pytest.mark.parametrize(
    "layout,heads,sparse_count,query_count,has_sinks",
    [
        ("PA_BSND", 1, 128, 257, False),
        ("PA_BSND", 12, 256, 3, False),
        ("PA_BSND", 64, 256, 3, False),
        ("PA_BSND", 65, 256, 3, False),
        ("PA_BSND", 128, 256, 3, False),
        ("TND", 12, 256, 3, False),
        ("BSND", 65, 256, 3, False),
        ("PA_BSND", 128, 2048, 3, False),
        ("PA_BSND", 65, 256, 3, True),
        ("PA_BSND", 128, 2048, 3, True),
        ("BSND", 12, 128, 3, True),
    ],
)
def test_dcp_empty_rows(layout, heads, sparse_count, query_count, has_sinks):
    kwargs, expected_attention, expected_lse, sinks = _inputs(
        layout, heads, sparse_count, query_count, torch.int8, torch.bfloat16, has_sinks
    )
    row_count = query_count * 2
    # Reuse the same shape across normal, empty and mixed requests to expose stale state.
    for pattern in ("valid", "empty", "alternating", "prefix", "suffix", "valid"):
        empty = torch.zeros(row_count, dtype=torch.bool)
        if pattern == "empty":
            empty[:] = True
        elif pattern == "alternating":
            empty[::2] = True
        elif pattern == "prefix":
            empty[: row_count // 2] = True
        elif pattern == "suffix":
            empty[row_count // 2 :] = True
        indices = (
            torch.arange(sparse_count, dtype=torch.int32)
            .expand(row_count, 1, -1)
            .clone()
        )
        indices[empty] = -1
        kwargs["sparse_indices"] = indices.reshape(kwargs["sparse_indices"].shape).npu()
        attention, maximum, total = kv_quant_sparse_flash_attention(**kwargs)
        torch.npu.synchronize()
        attention = attention.cpu().float().reshape(row_count, heads, 512)
        maximum = maximum.cpu().reshape(row_count, heads)
        total = total.cpu().reshape(row_count, heads)
        assert torch.isfinite(attention).all(), pattern
        assert torch.count_nonzero(attention[empty]) == 0, pattern
        if sinks is None:
            assert torch.all(
                maximum[empty] == torch.tensor(-2e38, dtype=torch.float32)
            ), pattern
            assert torch.all(total[empty] == 1), pattern
            empty_lse = maximum[empty] + total[empty].log()
            assert torch.isfinite(empty_lse).all(), pattern
            assert torch.count_nonzero(empty_lse.exp()) == 0, pattern
            assert (
                torch.count_nonzero(attention[empty] * empty_lse.exp().unsqueeze(-1))
                == 0
            ), pattern
        else:
            torch.testing.assert_close(
                maximum[empty], sinks.expand(int(empty.sum()), -1), rtol=0, atol=0
            )
            assert torch.all(total[empty] == 1), pattern
        valid_count = int((~empty).sum())
        if valid_count:
            torch.testing.assert_close(
                attention[~empty],
                expected_attention.expand(valid_count, -1, -1),
                atol=0.01,
                rtol=0.01,
            )
            torch.testing.assert_close(
                maximum[~empty] + total[~empty].log(),
                expected_lse.expand(valid_count, -1),
                atol=0.01,
                rtol=0.01,
            )


@pytest.mark.ci
@pytest.mark.parametrize("query_dtype", [torch.float16, torch.bfloat16])
def test_dcp_empty_without_statistics(query_dtype):
    kwargs, _, _, _ = _inputs("PA_BSND", 65, 256, 3, torch.int8, query_dtype, False)
    kwargs["sparse_indices"].fill_(-1)
    kwargs["return_softmax_lse"] = False
    attention, maximum, total = kv_quant_sparse_flash_attention(**kwargs)
    torch.npu.synchronize()
    assert torch.count_nonzero(attention.cpu()) == 0
    assert maximum.numel() == 0
    assert total.numel() == 0


@pytest.mark.ci
@pytest.mark.parametrize("layout", ["PA_BSND", "BSND"])
def test_dcp_zero_local_length(layout):
    kwargs, _, _, _ = _inputs(layout, 64, 256, 3, torch.int8, torch.bfloat16, False)
    kwargs["actual_seq_lengths_kv"].zero_()
    kwargs["sparse_indices"].fill_(-1)
    attention, maximum, total = kv_quant_sparse_flash_attention(**kwargs)
    torch.npu.synchronize()
    assert torch.count_nonzero(attention.cpu()) == 0
    assert torch.all(maximum.cpu() == 0)
    assert torch.all(total.cpu() == 1)


@pytest.mark.ci
@pytest.mark.parametrize(
    "layout,heads",
    [("PA_BSND", 64), ("BSND", 65), ("TND", 128), ("PA_BSND", 128), ("BSND", 128)],
)
@pytest.mark.parametrize(
    "valid_count", [0, 1, 63, 127, 128, 129, 255, 511, 512, 513, 2047, 2048]
)
def test_dcp_empty_tail_preserves_valid_blocks(layout, heads, valid_count):
    kwargs, expected_attention, expected_lse, _ = _inputs(
        layout, heads, 2048, 3, torch.int8, torch.bfloat16, False, valid_count
    )
    kwargs["sparse_indices"][..., valid_count:] = -1
    attention, maximum, total = kv_quant_sparse_flash_attention(**kwargs)
    torch.npu.synchronize()
    torch.testing.assert_close(
        attention.cpu().float().reshape(-1, heads, 512),
        expected_attention.expand(6, -1, -1),
        rtol=0.01,
        atol=0.01,
    )
    lse = maximum.cpu().reshape(-1, heads) + total.cpu().reshape(-1, heads).log()
    torch.testing.assert_close(lse, expected_lse.expand(6, -1), rtol=0.01, atol=0.01)


@pytest.mark.ci
@pytest.mark.parametrize("layout", ["PA_BSND", "BSND", "TND"])
@pytest.mark.parametrize("heads", [12, 14, 48, 64, 96, 128])
@pytest.mark.parametrize("query_dtype", [torch.float16, torch.bfloat16])
def test_v1_shared_kernel_empty_rows(layout, heads, query_dtype):
    query_count = 3
    sparse_count = 2048
    kwargs, expected_attention, _, _ = _inputs(
        layout,
        heads,
        sparse_count,
        query_count,
        torch.int8,
        query_dtype,
        False,
        default_zero_sink=False,
    )
    kwargs.pop("sinks")
    kwargs.pop("return_softmax_lse")
    row_count = 2 * query_count
    for pattern in (
        "valid",
        "empty",
        "alternating",
        "prefix_1",
        "prefix_127",
        "prefix_129",
        "prefix_511",
        "prefix_513",
        "prefix_2047",
        "valid",
    ):
        empty = torch.zeros(row_count, dtype=torch.bool)
        if pattern == "empty":
            empty[:] = True
        elif pattern == "alternating":
            empty[::2] = True
        indices = (
            torch.arange(sparse_count, dtype=torch.int32)
            .expand(row_count, 1, -1)
            .clone()
        )
        indices[empty] = -1
        expected_current = expected_attention
        if pattern.startswith("prefix_"):
            prefix = int(pattern.removeprefix("prefix_"))
            indices[..., prefix:] = -1
            _, expected_current, _, _ = _inputs(
                layout,
                heads,
                sparse_count,
                query_count,
                torch.int8,
                query_dtype,
                False,
                valid_count=prefix,
                default_zero_sink=False,
            )
        kwargs["sparse_indices"] = indices.reshape(kwargs["sparse_indices"].shape).npu()
        attention = torch_npu.npu_kv_quant_sparse_flash_attention(**kwargs)
        torch.npu.synchronize()
        attention = attention.cpu().float().reshape(row_count, heads, 512)
        assert torch.isfinite(attention).all(), pattern
        assert torch.count_nonzero(attention[empty]) == 0, pattern
        valid_rows = int((~empty).sum())
        if valid_rows:
            torch.testing.assert_close(
                attention[~empty],
                expected_current.expand(valid_rows, -1, -1),
                atol=0.01,
                rtol=0.01,
            )


@pytest.mark.ci
@pytest.mark.parametrize(
    "layout,heads", [("BSND", 64), ("TND", 128), ("PA_BSND", 128), ("PA_BSND", 12)]
)
@pytest.mark.parametrize("rope_dim", [0, 64])
@pytest.mark.parametrize("return_statistics", [False, True])
def test_dcp_sparse_prefix_reuse(layout, heads, rope_dim, return_statistics):
    # Traceable values exercise changed AIV boundaries and the no-RoPE address cache.
    capacity = 2048
    kwargs, _, _, _ = _inputs(
        layout, heads, capacity, 3, torch.int8, torch.bfloat16, False
    )
    markers = (torch.arange(capacity, dtype=torch.int32) % 15 - 7).to(torch.int8)
    latent = markers[:, None].expand(-1, 512).contiguous()
    rope = torch.zeros(capacity, rope_dim, dtype=torch.bfloat16)
    scales = torch.ones(capacity, 4, dtype=torch.float32)
    packed = torch.cat(
        [latent.view(torch.uint8), rope.view(torch.uint8), scales.view(torch.uint8)],
        dim=-1,
    ).view(torch.int8)
    slot = 528 + rope_dim * 2
    if layout == "PA_BSND":
        key = packed.reshape(-1, 128, 1, slot)
    else:
        key = packed.reshape(1, capacity, 1, slot).expand(2, -1, -1, -1).contiguous()
        if layout == "TND":
            key = key.flatten(0, 1)
    kwargs["key"] = key.npu()
    kwargs["value"] = kwargs["key"]
    q_shape = (*kwargs["query"].shape[:-1], 512 + rope_dim)
    kwargs["query"] = torch.zeros(q_shape, dtype=torch.bfloat16).npu()
    kwargs["rope_head_dim"] = rope_dim
    kwargs["return_softmax_lse"] = return_statistics
    for valid_count in (
        2048,
        1,
        127,
        128,
        129,
        511,
        512,
        513,
        1023,
        1024,
        1025,
        2047,
        0,
        2048,
    ):
        indices = torch.arange(capacity, dtype=torch.int32).expand(6, 1, -1).clone()
        indices[..., valid_count:] = -1
        kwargs["sparse_indices"].copy_(
            indices.reshape(kwargs["sparse_indices"].shape).npu()
        )
        attention, maximum, total = kv_quant_sparse_flash_attention(**kwargs)
        torch.npu.synchronize()
        expected = markers[:valid_count].float().sum() / (valid_count + 1)
        expected = expected.to(torch.bfloat16).float()
        torch.testing.assert_close(
            attention.cpu().float(),
            expected.expand(attention.shape),
            rtol=0.01,
            atol=0.0001,
        )
        if return_statistics:
            assert torch.all(maximum.cpu() == 0), valid_count
            assert torch.all(total.cpu() == valid_count + 1), valid_count
        else:
            assert maximum.numel() == total.numel() == 0


@pytest.mark.ci
@pytest.mark.parametrize(
    "layout,heads", [("PA_BSND", 12), ("PA_BSND", 96), ("TND", 48), ("BSND", 128)]
)
@pytest.mark.parametrize("has_sinks", [False, True])
def test_dcp_causal_masked_rows_preserve_sink(layout, heads, has_sinks):
    kwargs, _, _, sinks = _inputs(
        layout, heads, 256, 3, torch.int8, torch.bfloat16, has_sinks
    )
    rows = torch.full((6, 1, 256), -1, dtype=torch.int32)
    for row in range(6):
        threshold = 254 + row % 3
        future = torch.arange(threshold, 256, dtype=torch.int32)
        rows[row, 0, : future.numel()] = future
    kwargs["sparse_indices"] = rows.reshape(kwargs["sparse_indices"].shape).npu()
    kwargs["sparse_mode"] = 3
    attention, maximum, total = kv_quant_sparse_flash_attention(**kwargs)
    torch.npu.synchronize()
    assert torch.count_nonzero(attention.cpu()) == 0
    torch.testing.assert_close(
        maximum.cpu().reshape(6, heads), sinks.expand(6, -1), rtol=0, atol=0
    )
    assert torch.all(total.cpu() == 1)
