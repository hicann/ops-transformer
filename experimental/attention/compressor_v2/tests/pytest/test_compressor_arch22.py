# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import contextlib
import io

import pytest
import torch
import torch_npu

from cann_ops_transformer_custom.ops.attention.compressor_v2 import compressor
from compressor_golden_arch22 import check_result, cpu_compressor


@pytest.mark.ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout_th", [False, True])
@pytest.mark.parametrize("length", [1, 4, 8, 1003])
@pytest.mark.parametrize("head_dim", [128, 512])
def test_arch22_ring_state(dtype, layout_th, length, head_dim):
    _run_ring_sequence(dtype, layout_th, [length, 1, 1, 1, 1], head_dim)


@pytest.mark.ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("start_pos", [5, 6, 7])
@pytest.mark.parametrize("length", [3, 5])
def test_arch22_read_before_write_conflict(dtype, length, start_pos):
    _run_ring_sequence(
        dtype,
        False,
        [length],
        head_dim=128,
        capacity=3,
        starts=[start_pos],
        state_padding=1,
    )


@pytest.mark.ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_arch22_full_load_empty_loop_advances_batch_cursor(dtype):
    starts = [0, 0, 1] + [0] * 78
    _run_ring_sequence(
        dtype,
        False,
        [3],
        head_dim=512,
        capacity=8,
        starts=starts,
    )


def _run_ring_sequence(
    dtype,
    layout_th,
    lengths,
    head_dim=128,
    hidden=1024,
    capacity=1024,
    ratio=4,
    starts=None,
    used=None,
    th_lengths=None,
    state_padding=1,
):
    torch_npu.npu.set_device(0)
    torch.manual_seed(7317)
    starts = torch.tensor([1021, 2046] if starts is None else starts, dtype=torch.int32)
    batch = len(starts)
    wkv = torch.randn(head_dim, hidden).to(dtype) * 0.1
    wgate = torch.randn(head_dim, hidden).to(dtype) * 0.1
    device_wkv, device_wgate = wkv.npu(), wgate.npu()
    table = torch.arange(batch - 1, -1, -1, dtype=torch.int32)
    device_table = table.npu()
    reference = torch.randn(batch, capacity, 2 * head_dim) * 0.1
    backing = torch.full(
        (batch, capacity + state_padding, 2 * head_dim), 17.0, device="npu"
    )
    state = backing[:, :capacity]
    state.copy_(reference)
    for step_length in lengths:
        x = torch.randn(batch, step_length, hidden).to(dtype) * 0.1
        batch_lengths = torch.tensor(
            [step_length] * batch if th_lengths is None else th_lengths,
            dtype=torch.int32,
        )
        cumulative = (
            torch.cat(
                (torch.zeros(1, dtype=torch.int32), batch_lengths.cumsum(0))
            ).int()
            if layout_th
            else None
        )
        seq_used = None if used is None else torch.tensor(used, dtype=torch.int32)
        if layout_th:
            x = torch.cat(
                [
                    x[batch_idx, : int(batch_lengths[batch_idx])]
                    for batch_idx in range(batch)
                ]
            )
        kv = reference[:, :, :head_dim].contiguous()
        score = reference[:, :, head_dim:].contiguous()
        update_kv = torch.zeros_like(kv, dtype=torch.bool)
        update_score = torch.zeros_like(score, dtype=torch.bool)
        with contextlib.redirect_stdout(io.StringIO()):
            expected, valid, *_ = cpu_compressor(
                x,
                wkv,
                wgate,
                kv,
                score,
                update_kv,
                update_score,
                block_table=table,
                cu_seqlens=cumulative,
                start_pos=starts,
                cmp_ratio=ratio,
                seqused=seq_used,
            )
        before = state.cpu().clone()
        actual = compressor(
            x.npu(),
            device_wkv,
            device_wgate,
            state,
            cmp_ratio=ratio,
            state_block_table=device_table,
            start_pos=starts.npu(),
            cu_seqlens=None if cumulative is None else cumulative.npu(),
            seqused=None if seq_used is None else seq_used.npu(),
        ).cpu()
        if expected[valid].numel():
            with contextlib.redirect_stdout(io.StringIO()) as capture:
                _, verdict = check_result(
                    expected[valid].float(),
                    actual[valid].float(),
                    str(dtype).split(".")[-1],
                )
            assert verdict == "Pass", capture.getvalue()
        reference = torch.cat((kv, score), dim=-1)
        actual_state = state.cpu()
        torch.testing.assert_close(
            actual_state,
            reference,
            rtol=1e-4,
            atol=1e-5,
            msg=lambda detail: f"starts={starts.tolist()}, length={step_length}\n{detail}",
        )
        updated = torch.cat((update_kv, update_score), dim=-1)
        assert torch.equal(actual_state[~updated], before[~updated])
        assert torch.all(backing[:, capacity:].cpu() == 17)
        starts += batch_lengths if seq_used is None else seq_used


@pytest.mark.ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout_th", [False, True])
@pytest.mark.parametrize("ratio", [2, 4, 8, 16, 32, 64, 128])
@pytest.mark.parametrize("length", [1, 8])
def test_arch22_ratios(dtype, layout_th, ratio, length):
    _run_ring_sequence(
        dtype,
        layout_th,
        [length, 1],
        ratio=ratio,
        capacity=ratio + length - 1,
        starts=[ratio - 1, 2 * ratio - 1],
    )


@pytest.mark.ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout_th", [False, True])
@pytest.mark.parametrize(
    "scenario", ["empty", "unused", "partial", "multiple_wraps", "split_k_control"]
)
def test_arch22_boundaries(dtype, layout_th, scenario):
    if scenario == "empty":
        _run_ring_sequence(dtype, layout_th, [0])
    elif scenario == "unused":
        _run_ring_sequence(dtype, layout_th, [8], used=[0, 0])
    elif scenario == "partial":
        _run_ring_sequence(dtype, layout_th, [8, 8], used=[0, 5])
    elif scenario == "multiple_wraps":
        _run_ring_sequence(dtype, layout_th, [1] * 20, capacity=4, starts=[3, 6])
    else:
        _run_ring_sequence(dtype, layout_th, [8, 1, 1], hidden=4096)


@pytest.mark.ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("batch_lengths", [[0, 8, 4], [8, 0, 4], [8, 4, 0]])
def test_arch22_th_empty_batches(dtype, batch_lengths):
    _run_ring_sequence(
        dtype, True, [8], starts=[1021, 2046, 1023], th_lengths=batch_lengths
    )


@pytest.mark.ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout_th", [False, True])
def test_arch22_contiguous_state(dtype, layout_th):
    _run_ring_sequence(dtype, layout_th, [1] * 5, state_padding=0)
