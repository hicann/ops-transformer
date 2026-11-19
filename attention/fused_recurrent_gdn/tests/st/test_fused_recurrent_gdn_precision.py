# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Numerical checks through the installed Torch wheel on the NPU."""

import math
from pathlib import Path
import sys

import pytest
import torch
import torch_npu  # noqa: F401

from cann_ops_transformer_custom.ops.attention.fused_recurrent_gdn import (
    fused_recurrent_gdn,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from assets.golden import recurrent_gdn  # noqa: E402
from assets.inputs import activation_kwargs, make_inputs  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.npu.is_available(), reason="requires an NPU")


def _compare(label, actual, reference, *, rtol, atol):
    actual = actual.cpu().double()
    reference = reference.double()
    error = (actual - reference).abs()
    bad = ~torch.isclose(actual, reference, rtol=rtol, atol=atol)
    first = bad.nonzero()[0].tolist() if bad.any() else None
    print(
        f"{label}: max_abs={error.max().item():.9g} failures={bad.sum().item()} first={first}"
    )
    torch.testing.assert_close(actual, reference, rtol=rtol, atol=atol)


def _check_golden(case, inputs, state, *, scale=None):
    device_inputs = tuple(tensor.npu() for tensor in inputs)
    device_state = None if state is None else state.npu()
    options = activation_kwargs(
        inputs[3].dtype, inputs[4].dtype, inputs[2].shape[1], device="npu"
    )
    out, final = fused_recurrent_gdn(
        *device_inputs,
        device_state,
        scale=scale,
        output_final_state=True,
        **options,
    )
    torch.npu.synchronize()
    expected_out, expected_final = recurrent_gdn(
        *inputs,
        state,
        scale,
        **activation_kwargs(inputs[3].dtype, inputs[4].dtype, inputs[2].shape[1]),
    )
    assert out.dtype == torch.bfloat16 and final.dtype == torch.float32
    _compare(case + " out", out, expected_out.bfloat16(), rtol=1e-2, atol=2e-3)
    _compare(case + " state", final, expected_final, rtol=1e-4, atol=1e-5)
    if state is not None:
        torch.testing.assert_close(device_state.cpu(), state, rtol=0, atol=0)
    return out, final, device_inputs, device_state


@pytest.mark.parametrize("tokens", (1, 2, 4))
@pytest.mark.parametrize("heads", (1, 2))
@pytest.mark.parametrize("beta_dtype", (torch.bfloat16, torch.float32))
@pytest.mark.parametrize("gate_dtype", (torch.bfloat16, torch.float32))
def test_dtype_and_shape_grid(tokens, heads, beta_dtype, gate_dtype):
    inputs, state = make_inputs(tokens, heads, beta_dtype, gate_dtype)
    _check_golden(
        f"T={tokens} Nv={heads} beta={beta_dtype} g={gate_dtype}", inputs, state
    )


def test_zero_state_sparse_update():
    q = torch.zeros((1, 1, 128), dtype=torch.bfloat16)
    k, v = q.clone(), q.clone()
    q[0, 0, 3] = k[0, 0, 3] = 1
    v[0, 0, 7] = 2
    out, final, _, _ = _check_golden(
        "zero-state",
        (q, k, v, torch.tensor([[0.5]]), torch.zeros((1, 1))),
        None,
        scale=0.25,
    )
    expected = torch.zeros_like(final.cpu())
    expected[0, 0, 3, 7] = 1
    torch.testing.assert_close(final.cpu(), expected, rtol=0, atol=0)
    expected_out = torch.zeros_like(v)
    expected_out[0, 0, 7] = 0.25
    torch.testing.assert_close(out.cpu(), expected_out, rtol=0, atol=0)


@pytest.mark.parametrize("write_scale", (0.0, 0.5))
def test_decay_before_prediction(write_scale):
    q = torch.zeros((1, 1, 128), dtype=torch.bfloat16)
    k, v = q.clone(), q.clone()
    q[0, 0, 3] = k[0, 0, 3] = 1
    v[0, 0, 7] = 1
    state = torch.zeros((1, 1, 128, 128))
    state[0, 0, 3, 7] = 4
    inputs = (q, k, v, torch.tensor([[write_scale]]), torch.tensor([[-math.log(2)]]))
    out, final, _, _ = _check_golden("decay", inputs, state, scale=1)
    expected = 2 - write_scale
    assert final[0, 0, 3, 7].item() == pytest.approx(expected, abs=1e-6)
    assert out[0, 0, 7].item() == expected


def test_segmented_scale_and_optional_state():
    inputs, state = make_inputs(4, 2, torch.float32, torch.bfloat16)
    out, final, npu_inputs, npu_state = _check_golden(
        "segmented baseline", inputs, state
    )
    options = activation_kwargs(inputs[3].dtype, inputs[4].dtype, 2, device="npu")
    parts = []
    step_state = npu_state
    for token in range(4):
        part, step_state = fused_recurrent_gdn(
            *(tensor[token : token + 1] for tensor in npu_inputs),
            step_state,
            output_final_state=True,
            **options,
        )
        parts.append(part)
    torch.npu.synchronize()
    _compare("segmented out", torch.cat(parts), out.cpu(), rtol=0, atol=0)
    _compare("segmented state", step_state, final.cpu(), rtol=0, atol=0)

    scale = 2 * 128**-0.5
    scaled, scaled_state = fused_recurrent_gdn(
        *npu_inputs,
        npu_state,
        scale=scale,
        output_final_state=True,
        **options,
    )
    optional, absent = fused_recurrent_gdn(
        *npu_inputs, npu_state, scale=scale, **options
    )
    torch.npu.synchronize()
    expected, _ = recurrent_gdn(
        *inputs, state, scale, **activation_kwargs(inputs[3].dtype, inputs[4].dtype, 2)
    )
    _compare("explicit scale out", scaled, expected.bfloat16(), rtol=1e-2, atol=2e-3)
    _compare("scale does not change state", scaled_state, final.cpu(), rtol=0, atol=0)
    _compare("optional state out", optional, scaled.cpu(), rtol=0, atol=0)
    assert absent is None


def test_fp32_gate_and_beta_are_not_rounded_to_bf16():
    inputs, state = make_inputs(4, 2, torch.float32, torch.float32, seed=731)
    _, final, _, _ = _check_golden("FP32 gate/beta", inputs, state)
    for index in (3, 4):
        rounded = list(inputs)
        rounded[index] = rounded[index].bfloat16().float()
        _, rounded_final = recurrent_gdn(*rounded, state)
        assert (final.cpu().double() - rounded_final).abs().max().item() > 1e-5


@pytest.mark.parametrize("layout", ("TND", "NTD", "BSND", "BNSD"))
@pytest.mark.parametrize("dim", (64, 65, 128, 256, 511, 512))
@pytest.mark.parametrize("raw", (False, True))
@pytest.mark.parametrize("state_dtype", (torch.bfloat16, torch.float32))
def test_extended_backend(layout, dim, raw, state_dtype):
    dtype = torch.bfloat16 if raw else torch.float32
    data, state = make_inputs(
        2,
        2,
        dtype,
        dtype,
        key_dim=dim,
        value_dim=576 - dim,
        batch=2,
        state_dtype=state_dtype,
        layout=layout,
    )
    opts = dict(
        layout=layout,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        allow_neg_eigval=raw,
    )
    if layout in ("TND", "NTD"):
        opts["cu_seqlens"] = torch.tensor([0, 2, 4], dtype=torch.int32)
    expected, expected_state = recurrent_gdn(
        *data, state, **opts, **activation_kwargs(dtype, dtype, 2)
    )
    device_opts = {
        k: v.npu() if isinstance(v, torch.Tensor) else v for k, v in opts.items()
    }
    device_state = state.npu()
    actual, final = fused_recurrent_gdn(
        *(x.npu() for x in data),
        device_state,
        **device_opts,
        **activation_kwargs(dtype, dtype, 2, device="npu"),
    )
    torch.npu.synchronize()
    assert final.dtype == state_dtype
    _compare("extended out", actual, expected.bfloat16(), rtol=1e-2, atol=2e-3)
    _compare("extended state", final, expected_state, rtol=1e-2, atol=2e-3)
    torch.testing.assert_close(device_state.cpu(), state, rtol=0, atol=0)


@pytest.mark.parametrize("rank", (1, 2))
@pytest.mark.parametrize("state_dtype", (torch.bfloat16, torch.float32))
def test_state_pool_writeback(rank, state_dtype):
    data, _ = make_inputs(
        2, 2, torch.float32, torch.float32, key_dim=64, value_dim=96, batch=2
    )
    state = torch.randn(8, 2, 64, 96).mul_(0.02).to(state_dtype)
    indices = torch.tensor([1, 5] if rank == 1 else [[1, 2], [5, 6]], dtype=torch.int32)
    opts = dict(
        cu_seqlens=torch.tensor([0, 2, 4], dtype=torch.int32), ssm_state_indices=indices
    )
    if rank == 2:
        opts["num_accepted_tokens"] = torch.tensor([2, 1], dtype=torch.int32)
    reference_state = state.clone()
    expected, _ = recurrent_gdn(
        *data, reference_state, output_final_state=False, **opts
    )
    # A noncontiguous pool exercises copy-back to the caller's original storage.
    device_state = state.transpose(-1, -2).contiguous().transpose(-1, -2).npu()
    actual, final = fused_recurrent_gdn(
        *(x.npu() for x in data), device_state, **{k: v.npu() for k, v in opts.items()}
    )
    torch.npu.synchronize()
    assert final is None
    _compare("pool out", actual, expected.bfloat16(), rtol=1e-2, atol=2e-3)
    _compare("pool state", device_state, reference_state, rtol=1e-2, atol=2e-3)
    untouched = [0, 3, 4, 7] if rank == 2 else [0, 2, 3, 4, 6, 7]
    torch.testing.assert_close(
        device_state.cpu()[untouched], state[untouched], rtol=0, atol=0
    )


@pytest.mark.parametrize("layout", ("TND", "NTD", "BSND", "BNSD"))
def test_ragged_native_pipeline(layout):
    data, state = make_inputs(
        4,
        2,
        torch.bfloat16,
        torch.float32,
        key_dim=64,
        value_dim=96,
        batch=3,
        state_dtype=torch.bfloat16,
        layout=layout,
    )
    opts = dict(
        layout=layout,
        seqused=torch.tensor([0, 2, 3], dtype=torch.int32),
        output_final_state=True,
    )
    if layout in ("TND", "NTD"):
        opts["cu_seqlens"] = torch.tensor([0, 4, 8, 12], dtype=torch.int32)
    expected, expected_state = recurrent_gdn(
        *data, state, **opts, **activation_kwargs(torch.bfloat16, torch.float32, 2)
    )
    initial = state.npu()
    actual, final = fused_recurrent_gdn(
        *(x.npu() for x in data),
        initial,
        **{k: v.npu() if isinstance(v, torch.Tensor) else v for k, v in opts.items()},
        **activation_kwargs(torch.bfloat16, torch.float32, 2, device="npu"),
    )
    torch.npu.synchronize()
    _compare("ragged out", actual, expected.bfloat16(), rtol=1e-2, atol=2e-3)
    _compare("ragged state", final, expected_state, rtol=1e-2, atol=2e-3)
    torch.testing.assert_close(final.cpu()[0], state[0], rtol=0, atol=0)
    torch.testing.assert_close(initial.cpu(), state, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ("TND", "NTD", "BSND", "BNSD"))
def test_gate_parameters_and_noncontiguous_input(layout):
    data, state = make_inputs(
        3, 2, torch.bfloat16, torch.bfloat16, key_dim=64, value_dim=96, layout=layout
    )
    # Preserve logical values while creating strided source tensors.
    data = tuple(x.transpose(-1, -2).contiguous().transpose(-1, -2) for x in data)
    options = dict(
        layout=layout,
        output_final_state=True,
        use_beta_sigmoid_in_kernel=True,
        use_gate_in_kernel=True,
        allow_neg_eigval=True,
        A_log=torch.tensor([0.7, -0.7]),
        dt_bias=torch.tensor([10.0, -10.0]),
    )
    expected, expected_state = recurrent_gdn(*data, state, **options)
    actual, final = fused_recurrent_gdn(
        *(x.npu() for x in data),
        state.npu(),
        **{
            k: v.npu() if isinstance(v, torch.Tensor) else v for k, v in options.items()
        },
    )
    torch.npu.synchronize()
    _compare("gate out", actual, expected.bfloat16(), rtol=1e-2, atol=2e-3)
    _compare("gate state", final, expected_state, rtol=1e-4, atol=1e-5)
