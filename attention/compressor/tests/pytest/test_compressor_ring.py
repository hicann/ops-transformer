# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import contextlib
import io
import os

import pytest
import torch
import torch_npu

from cann_ops_transformer.ops import compressor_v2
from compressor_golden import check_result
from compressor_ring_golden import cpu_compressor_ring


def _run_ring_sequence(
    dtype,
    layout_th,
    lengths,
    *,
    head_dim=128,
    hidden=1024,
    capacity=1024,
    ratio=4,
    starts=None,
    ape_mode="nonzero",
    state_padding=1,
):
    """Run a ring-buffer sequence and validate output, updated rows, and padding."""
    state_padding = int(os.getenv("COMPRESSOR_TEST_STATE_PADDING", state_padding))
    torch_npu.npu.set_device(0)
    torch.manual_seed(7317)
    starts = torch.tensor(
        [capacity - 3, 2 * capacity - 2] if starts is None else starts,
        dtype=torch.int32,
    )
    batch = len(starts)
    wkv = torch.randn(head_dim, hidden).to(dtype) * 0.1
    wgate = torch.randn(head_dim, hidden).to(dtype) * 0.1
    device_wkv, device_wgate = wkv.npu(), wgate.npu()
    # Reverse IDs include physical block 0, which is a valid ring-buffer slot.
    table = torch.arange(batch - 1, -1, -1, dtype=torch.int32)
    ape = torch.randn(ratio, head_dim, dtype=torch.float32) * 0.1
    ape_reference = torch.zeros_like(ape) if ape_mode == "none" else ape

    reference = torch.randn(batch, capacity, 2 * head_dim) * 0.1
    backing = torch.full(
        (batch, capacity + state_padding, 2 * head_dim),
        17.0,
        device="npu",
    )
    state = backing[:, :capacity]
    state.copy_(reference)

    for step_length in lengths:
        x = torch.randn(batch, step_length, hidden).to(dtype) * 0.1
        if layout_th:
            x = x.reshape(batch * step_length, hidden)
            cumulative = torch.arange(
                0, (batch + 1) * step_length, step_length, dtype=torch.int32
            )
        else:
            cumulative = None

        reference_kv = reference[:, :, :head_dim].contiguous()
        reference_score = reference[:, :, head_dim:].contiguous()
        update_kv = torch.zeros_like(reference_kv, dtype=torch.bool)
        update_score = torch.zeros_like(reference_score, dtype=torch.bool)
        with contextlib.redirect_stdout(io.StringIO()):
            expected, valid, *_ = cpu_compressor_ring(
                x,
                wkv,
                wgate,
                reference_kv,
                reference_score,
                update_kv,
                update_score,
                ape_reference,
                block_table=table,
                cu_seqlens=cumulative,
                start_pos=starts,
                cmp_ratio=ratio,
                coff=1,
                cache_mode=2,
            )

        before = state.cpu().clone()
        actual = compressor_v2(
            x.npu(),
            device_wkv,
            device_wgate,
            state,
            ratio,
            ape=None if ape_mode == "none" else ape.npu(),
            state_block_table=table.npu(),
            cu_seqlens=None if cumulative is None else cumulative.npu(),
            start_pos=starts.npu(),
            coff=1,
            cache_mode=2,
        ).cpu()

        if expected[valid].numel():
            with contextlib.redirect_stdout(io.StringIO()) as capture:
                _, verdict = check_result(
                    expected[valid].float(),
                    actual[valid].float(),
                    str(dtype).split(".")[-1],
                )
            assert verdict == "Pass", capture.getvalue()

        actual_state = state.cpu()
        actual_kv, actual_score = (
            actual_state[..., :head_dim],
            actual_state[..., head_dim:],
        )
        torch.testing.assert_close(
            actual_kv[update_kv],
            reference_kv[update_kv],
            rtol=1e-4,
            atol=1e-5,
        )
        torch.testing.assert_close(
            actual_score[update_score],
            reference_score[update_score],
            rtol=1e-4,
            atol=1e-5,
        )
        updated = torch.cat((update_kv, update_score), dim=-1)
        assert torch.equal(actual_state[~updated], before[~updated])
        if state_padding:
            assert torch.equal(
                backing[:, capacity:].cpu(),
                torch.full_like(backing[:, capacity:].cpu(), 17),
            )
        reference = torch.cat((reference_kv, reference_score), dim=-1)
        starts += step_length


@pytest.mark.ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout_th", [False, True], ids=["full_load_bsh", "normal_th"])
@pytest.mark.parametrize("ape_mode", ["none", "nonzero"])
def test_ring_ape_layout_and_template(dtype, layout_th, ape_mode):
    _run_ring_sequence(
        dtype,
        layout_th,
        [4 if not layout_th else 5],
        capacity=8,
        starts=[7, 15],
        ape_mode=ape_mode,
    )


@pytest.mark.ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("layout_th", [False, True])
def test_ring_snapshot_wrap_and_block_zero(dtype, layout_th):
    _run_ring_sequence(
        dtype,
        layout_th,
        [5],
        capacity=8,
        starts=[7, 15],
        ape_mode="nonzero",
        state_padding=1,
    )


@pytest.mark.ci
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_ring_multiple_wrap_iterations(dtype):
    _run_ring_sequence(
        dtype,
        True,
        [17, 1, 1],
        capacity=8,
        starts=[7, 15],
        ape_mode="none",
        state_padding=1,
    )


@pytest.mark.ci
def test_ring_cache_mode_rejects_coff2():
    torch_npu.npu.set_device(0)
    batch, seq_len, hidden, head_dim, ratio, capacity = 1, 4, 1024, 128, 4, 8
    x = torch.randn(batch, seq_len, hidden, dtype=torch.float16, device="npu")
    wkv = torch.randn(2 * head_dim, hidden, dtype=torch.float16, device="npu")
    wgate = torch.randn(2 * head_dim, hidden, dtype=torch.float16, device="npu")
    state = torch.randn(
        batch, capacity, 4 * head_dim, dtype=torch.float32, device="npu"
    )
    ape = torch.randn(ratio, 2 * head_dim, dtype=torch.float32, device="npu")
    with pytest.raises(RuntimeError):
        compressor_v2(
            x,
            wkv,
            wgate,
            state,
            ratio,
            ape=ape,
            state_block_table=torch.zeros(batch, dtype=torch.int32, device="npu"),
            start_pos=torch.full(
                (batch,), capacity - 1, dtype=torch.int32, device="npu"
            ),
            coff=2,
            cache_mode=2,
        )
