# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PR review regressions: FP32 tails/intervals and optional-state resources."""

import importlib
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from cann_ops_transformer import flash_gdn, flash_gdn_fwd, flash_gdn_metadata

TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TESTS / "assets" / "impl"))
import gdn_ttk_ops


def _kernel_module():
    return importlib.import_module(
        "cann_ops_transformer.ops.attention.flash_gdn.op_dsl.flash_gdn_kernel"
    )


def test_softplus_source_formula_preserves_small_tail(monkeypatch):
    # Replay this exact scalar helper with FP32 register operations only.
    # This is a formula unit test, not an operator backend or NPU substitute.
    def f32(value):
        return np.asarray(value, dtype=np.float32)

    def binary(function):
        def operation(left, right, **_kwargs):
            return f32(function(f32(left), f32(right)))

        return operation

    def unary(function):
        def operation(value, **_kwargs):
            return f32(function(f32(value)))

        return operation

    kernel = _kernel_module()
    operations = dict(
        dtypes=SimpleNamespace(float32=np.float32),
        vdups=lambda value, dtype: f32(value),
        vexp=unary(np.exp),
        vabs=unary(np.abs),
        vmuls=binary(np.multiply),
        vmul=binary(np.multiply),
        vdiv=binary(np.divide),
        vadds=binary(np.add),
        vadd=binary(np.add),
        vmaxs=binary(np.maximum),
    )
    for name, operation in operations.items():
        monkeypatch.setattr(kernel, name, operation)
    values = np.array(
        [
            -80.0,
            -30.0,
            -20.0,
            -17.0,
            -16.0,
            -10.0,
            -1.0,
            -1e-6,
            0.0,
            1e-6,
            1.0,
            10.0,
            20.0,
            80.0,
        ],
        dtype=np.float32,
    )
    result = kernel._softplus_stable(values, None)
    expected = np.logaddexp(0.0, values.astype(np.float64))
    np.testing.assert_allclose(result, expected, rtol=5e-7, atol=0.0)
    assert result[2] > 0
    np.testing.assert_allclose(np.exp(np.float32(20.0)) * result[2], 1.0, rtol=5e-7)


@pytest.mark.parametrize("length", [2, 64])
def test_independent_fp32_peer_retains_gate_intervals(length):
    q = torch.zeros((1, length, 128))
    q[..., 0] = 1
    v = q.clone() * 4
    v[:, 0, 0] = 2
    g = torch.full((1, length), -1.0)
    g[:, 0] = -1e8
    beta = torch.full_like(g, 0.5)
    state = torch.zeros((1, 128, 128))
    out, final = gdn_ttk_ops.load_asset("peer")._chunk(q, q, v, g, beta, state, 1.0)
    expected = []
    value = 0.0
    for token in range(length):
        value *= float(np.exp(float(g[0, token])))
        value += 0.5 * (float(v[0, token, 0]) - value)
        expected.append(value)
    torch.testing.assert_close(
        out[0, :, 0], torch.tensor(expected), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        final[0, 0, 0], torch.tensor(expected[-1]), atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize("save_final", [False, True])
def test_final_state_host_allocation_is_optional(monkeypatch, save_final):
    kernel = _kernel_module()
    calls = []
    monkeypatch.setattr(kernel, "_compiled", lambda: lambda *args: calls.append(args))
    q = torch.zeros((2, 1, 128), dtype=torch.bfloat16)
    g = torch.zeros((2, 1))
    header = [0, 0, 1, 1, 1, 2, 128, 128, 0, 0, 0, 0, 2]
    _, final, _ = kernel.run_device(
        q,
        q,
        q,
        g,
        g,
        None,
        1.0,
        "TND",
        header,
        None,
        None,
        False,
        False,
        False,
        False,
        False,
        save_final,
    )
    assert len(calls) == 1
    assert calls[0][14].numel() == (128 * 128 if save_final else 1)
    assert calls[0][26] == int(save_final)
    assert (final is not None) == save_final


NPU = pytest.mark.skipif(
    os.environ.get("RUN_GDN_NPU") != "1",
    reason="set RUN_GDN_NPU=1 for real NPU regressions",
)


def _inputs(length, layout, *, softplus=False):
    q = torch.zeros((1, length, 1, 128), dtype=torch.bfloat16)
    q[..., 0] = 1
    k = torch.zeros_like(q) if softplus else q.clone()
    v = torch.zeros_like(q)
    if not softplus:
        v[..., 0] = 4
        v[:, 0, :, 0] = 2
    g = torch.full((1, length, 1), -20.25 if softplus else -1.0)
    if not softplus:
        g[:, 0] = -1e8
    beta = torch.zeros_like(g) if softplus else torch.full_like(g, 0.5)
    state = torch.zeros((1, 1, 128, 128))
    if softplus:
        state[0, 0, 0, 0] = 1

    def logical(value):
        if layout == "TND":
            return value[0].contiguous()
        return value.transpose(1, 2).contiguous() if layout == "BNSD" else value

    attributes = dict(
        layout=layout,
        valid_lengths=[length],
        initial_state_mode="random",
        dt_bias_mode="random" if softplus else "none",
        output_final_state=True,
        return_intermediate_states=True,
        scale=1.0,
        use_gate_in_kernel=softplus,
    )
    inputs = [logical(x) for x in (q, k, v, g, beta)] + [
        None,
        state,
        torch.tensor([20.0]),
        torch.tensor([0.25]),
        torch.tensor([0, length], dtype=torch.int32),
        torch.tensor([length], dtype=torch.int32),
    ]
    return inputs, attributes


def _device_call(function, inputs, attributes, *, final=True, chunks=True):
    device = os.environ.get("GDN_DEVICE", "npu:0")
    values = [x.to(device) if isinstance(x, torch.Tensor) else x for x in inputs[:9]]
    values[5] = flash_gdn_metadata(
        batch_size=1,
        nheads_qk=1,
        nheads_v=1,
        seqlens=attributes["valid_lengths"][0],
        layout=attributes["layout"],
    )
    return function(
        *values[:6],
        initial_state=values[6],
        A_log=values[7] if attributes["use_gate_in_kernel"] else None,
        dt_bias=values[8] if attributes["use_gate_in_kernel"] else None,
        layout=attributes["layout"],
        scale=1.0,
        output_final_state=final,
        return_intermediate_states=chunks,
        use_gate_in_kernel=attributes["use_gate_in_kernel"],
    )


@pytest.mark.npu
@NPU
@pytest.mark.parametrize("layout", ["TND", "BSND", "BNSD"])
@pytest.mark.parametrize("entry", ["flash_gdn", "flash_gdn_fwd"])
@pytest.mark.parametrize("length", [1, 2, 65])
def test_fused_softplus_tail_real_dispatch(layout, entry, length):
    import torch_npu  # noqa: F401

    torch.npu.set_device(os.environ.get("GDN_DEVICE", "npu:0"))
    values, attributes = _inputs(length, layout, softplus=True)
    expected = gdn_ttk_ops.load_asset("golden").reference(values, attributes)
    function = flash_gdn if entry == "flash_gdn" else flash_gdn_fwd
    actual = _device_call(function, values, attributes)
    torch.npu.synchronize()
    torch.testing.assert_close(actual[0].cpu(), expected[0], atol=0.0, rtol=0.0)
    for got, want in zip(actual[1:3], expected[1:3]):
        torch.testing.assert_close(got.cpu(), want, atol=1e-7, rtol=2e-5)
    assert float(actual[0].flatten()[0]) < 0.5


@pytest.mark.npu
@NPU
@pytest.mark.parametrize("layout", ["TND", "BSND", "BNSD"])
@pytest.mark.parametrize("length", [2, 65, 129])
def test_large_negative_gate_preserves_later_decay(layout, length):
    import torch_npu  # noqa: F401

    torch.npu.set_device(os.environ.get("GDN_DEVICE", "npu:0"))
    values, attributes = _inputs(length, layout)
    expected = gdn_ttk_ops.load_asset("golden").reference(values, attributes)
    actual = _device_call(flash_gdn_fwd, values, attributes)
    torch.npu.synchronize()
    torch.testing.assert_close(actual[0].cpu(), expected[0], atol=0.0, rtol=0.0)
    for got, want in zip(actual[1:3], expected[1:3]):
        torch.testing.assert_close(got.cpu(), want, atol=2e-6, rtol=2e-5)
    assert float(actual[1][0, 0, 0, 0]) < 2.5


@pytest.mark.npu
@NPU
@pytest.mark.parametrize("length", [2, 65])
@pytest.mark.parametrize("save_chunks", [False, True])
def test_final_state_disabled_preserves_out_and_does_not_write(
    monkeypatch, length, save_chunks
):
    import torch_npu  # noqa: F401

    torch.npu.set_device(os.environ.get("GDN_DEVICE", "npu:0"))
    kernel = _kernel_module()
    compiled = kernel._compiled()
    launches = []

    def measured(*args):
        # Delegate to the real compiled kernel; only the unused one-element
        # final-state buffer is initialized as a write canary.
        if not args[26]:
            assert args[14].numel() == 1
            args[14].fill_(1234.5)
        result = compiled(*args)
        torch.npu.synchronize()
        if not args[26]:
            assert args[14].cpu().tolist() == [1234.5]
        launches.append(bool(args[26]))
        return result

    monkeypatch.setattr(kernel, "_compiled", lambda: measured)
    values, attributes = _inputs(length, "BSND")
    on = _device_call(flash_gdn, values, attributes, final=True, chunks=save_chunks)
    off = _device_call(flash_gdn, values, attributes, final=False, chunks=save_chunks)
    assert on[1] is not None and off[1] is None
    assert torch.equal(on[0], off[0])
    if save_chunks:
        assert torch.equal(on[2], off[2])
    else:
        assert on[2] is off[2] is None
    assert launches == [True, False]
