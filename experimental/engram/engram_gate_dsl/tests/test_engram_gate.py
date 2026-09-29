# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import pytest
import torch

import pytest
import torch

import ops.engram_gate as _module

EngramGate = _module.EngramGate
EngramGateKernel = _module.EngramGateKernel
engram_gate = _module.engram_gate


def _golden(x, key, value, weight, eps, clamp_value, image_mask=None):
    x_f = x.float()
    rstd = torch.rsqrt(x_f.square().mean(-1) + eps) * torch.rsqrt(
        key.square().mean(-1) + eps
    )
    dot = (x_f * weight * key).sum(-1) * rstd * x.shape[-1] ** -0.5
    gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(clamp_value).sqrt(), dot))
    if image_mask is not None:
        gate = gate.masked_fill(image_mask.unsqueeze(-1), 0)
    return (x_f + gate.unsqueeze(-1) * value.float().unsqueeze(-2)).to(x.dtype)


def _inputs(t, hc, dim, seed):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn(t, hc, dim, generator=generator).to(torch.bfloat16)
    key = torch.randn(t, hc, dim, generator=generator).to(torch.bfloat16)
    value = torch.randn(t, dim, generator=generator).to(torch.bfloat16)
    weight = torch.randn(hc, dim, generator=generator, dtype=torch.float32)
    return x, key, value, weight


def test_engram_gate_compile_default_shape():
    pytest.importorskip("cannbotdsl")
    import cannbotdsl

    if torch.npu.is_available():
        max_blocks = torch.npu.get_device_properties(0).vector_core_num
    else:
        max_blocks = 8
    EngramGate(eps=1e-6, clamp_value=1e-6, max_blocks=max_blocks).run.compile(
        cannbotdsl.TensorSpec((2, 4, 5120), cannbotdsl.dtypes.bfloat16),
        cannbotdsl.TensorSpec((2, 4, 5120), cannbotdsl.dtypes.bfloat16),
        cannbotdsl.TensorSpec((2, 5120), cannbotdsl.dtypes.bfloat16),
        cannbotdsl.TensorSpec((4, 5120), cannbotdsl.dtypes.float32),
        cannbotdsl.TensorSpec((2, 4, 5120), cannbotdsl.dtypes.bfloat16),
    )


@pytest.mark.npu
@pytest.mark.parametrize(
    "t,mask_kind,eps,clamp_value",
    [
        pytest.param(1, "none", 1e-6, 1e-6, id="single_token"),
        pytest.param(5, "partial", 1e-6, 1e-6, id="partial_image_mask"),
        pytest.param(8, "all", 1e-5, 1e-6, id="all_image_tokens"),
        pytest.param(17, "none", 5e-7, 0.25, id="non_default_clamp"),
    ],
)
def test_engram_gate_npu_precision(t, mask_kind, eps, clamp_value):
    pytest.importorskip("torch_npu")
    x, key, value, weight = _inputs(t, 4, 5120, 20260920 + t)
    image_mask = None
    if mask_kind == "partial":
        image_mask = torch.arange(t) % 2 == 1
    elif mask_kind == "all":
        image_mask = torch.ones(t, dtype=torch.bool)
    expected = _golden(x, key, value, weight, eps, clamp_value, image_mask)
    actual = engram_gate(
        x.npu(),
        key.npu(),
        value.npu(),
        weight.npu(),
        None if image_mask is None else image_mask.npu(),
        eps=eps,
        clamp_value=clamp_value,
    )
    torch.npu.synchronize()
    actual_cpu = actual.cpu()
    max_error = (actual_cpu.float() - expected.float()).abs().max().item()
    assert torch.allclose(actual_cpu.float(), expected.float(), atol=2e-2, rtol=2e-2), (
        f"max absolute error: {max_error}"
    )
    if image_mask is not None:
        assert torch.equal(actual_cpu[image_mask], x[image_mask])


@pytest.mark.npu
@pytest.mark.parametrize(
    "lead,hc,dim,mask_kind",
    [
        pytest.param((5,), 1, 5120, "none", id="two_dim_hc1"),
        pytest.param((7,), 1, 127, "partial", id="two_dim_odd_dim_partial_mask"),
        pytest.param((2, 3), 4, 5120, "partial", id="four_dim_partial_mask"),
        pytest.param((2, 2, 2), 3, 64, "none", id="five_dim"),
    ],
)
def test_engram_gate_npu_precision_generalized_ndim(lead, hc, dim, mask_kind):
    pytest.importorskip("torch_npu")
    shape = lead + (hc, dim)
    total_tokens = 1
    for size in lead:
        total_tokens *= size
    generator = torch.Generator(device="cpu").manual_seed(20260922 + dim)
    x = torch.randn(*shape, generator=generator).to(torch.bfloat16)
    key = torch.randn(*shape, generator=generator).to(torch.bfloat16)
    value = torch.randn(*lead, dim, generator=generator).to(torch.bfloat16)
    weight = torch.randn(hc, dim, generator=generator, dtype=torch.float32)
    image_mask = None
    if mask_kind == "partial":
        image_mask = torch.arange(total_tokens).view(*lead) % 2 == 1
    expected = _golden(x, key, value, weight, 1e-6, 1e-6, image_mask)
    actual = engram_gate(
        x.npu(),
        key.npu(),
        value.npu(),
        weight.npu(),
        None if image_mask is None else image_mask.npu(),
    )
    torch.npu.synchronize()
    actual_cpu = actual.cpu()
    assert actual_cpu.shape == expected.shape
    max_error = (actual_cpu.float() - expected.float()).abs().max().item()
    assert torch.allclose(actual_cpu.float(), expected.float(), atol=2e-2, rtol=2e-2), (
        f"max absolute error: {max_error}"
    )
    if image_mask is not None:
        assert torch.equal(
            actual_cpu.view(total_tokens, hc, dim)[image_mask.view(-1)],
            x.view(total_tokens, hc, dim)[image_mask.view(-1)],
        )


@pytest.mark.npu
@pytest.mark.parametrize(
    "t,hc,dim,mask_kind",
    [
        pytest.param(29, 4, 5120, "none", id="pair_pipeline_phase_two"),
        pytest.param(72, 4, 5120, "none", id="model_token_count_nomask"),
        pytest.param(72, 4, 5120, "partial", id="model_token_count_partial_mask"),
        pytest.param(128, 4, 5120, "none", id="large_tokens_nomask"),
        pytest.param(100, 3, 5120, "partial", id="generic_mapping_partial_mask"),
    ],
)
def test_engram_gate_npu_precision_pair_pipeline(t, hc, dim, mask_kind):
    pytest.importorskip("torch_npu")
    x, key, value, weight = _inputs(t, hc, dim, 20260920 + t)
    image_mask = None
    if mask_kind == "partial":
        image_mask = torch.arange(t) % 2 == 1
    expected = _golden(x, key, value, weight, 1e-6, 1e-6, image_mask)
    actual = engram_gate(
        x.npu(),
        key.npu(),
        value.npu(),
        weight.npu(),
        None if image_mask is None else image_mask.npu(),
    )
    torch.npu.synchronize()
    actual_cpu = actual.cpu()
    max_error = (actual_cpu.float() - expected.float()).abs().max().item()
    assert torch.allclose(actual_cpu.float(), expected.float(), atol=2e-2, rtol=2e-2), (
        f"max absolute error: {max_error}"
    )
    if image_mask is not None:
        assert torch.equal(actual_cpu[image_mask], x[image_mask])


@pytest.mark.npu
@pytest.mark.parametrize(
    "t,hc,dim,mask_kind",
    [
        pytest.param(3, 4, 8449, "none", id="chunked_one_chunk_tail_one"),
        pytest.param(3, 4, 16385, "partial", id="chunked_two_chunks_tail_one"),
        pytest.param(2, 2, 40000, "partial", id="chunked_multi_chunk_aligned"),
        pytest.param(1, 4, 100000, "none", id="chunked_extreme_dim"),
    ],
)
def test_engram_gate_npu_precision_large_dim(t, hc, dim, mask_kind):
    pytest.importorskip("torch_npu")
    x, key, value, weight = _inputs(t, hc, dim, 20260923 + dim % 10000)
    image_mask = None
    if mask_kind == "partial":
        image_mask = torch.arange(t) % 2 == 1
    expected = _golden(x, key, value, weight, 1e-6, 1e-6, image_mask)
    actual = engram_gate(
        x.npu(),
        key.npu(),
        value.npu(),
        weight.npu(),
        None if image_mask is None else image_mask.npu(),
    )
    torch.npu.synchronize()
    actual_cpu = actual.cpu()
    max_error = (actual_cpu.float() - expected.float()).abs().max().item()
    assert torch.allclose(actual_cpu.float(), expected.float(), atol=2e-2, rtol=2e-2), (
        f"max absolute error: {max_error}"
    )
    if image_mask is not None:
        assert torch.equal(actual_cpu[image_mask], x[image_mask])


@pytest.mark.npu
@pytest.mark.parametrize(
    "t,hc,dim,mask_kind",
    [
        pytest.param(3, 4, 64, "none", id="single_full_segment"),
        pytest.param(3, 4, 65, "none", id="tail_one"),
        pytest.param(5, 4, 127, "partial", id="sub_128_odd_partial_mask"),
        pytest.param(17, 2, 5121, "none", id="model_dim_plus_one"),
        pytest.param(5, 4, 5121, "partial", id="model_dim_plus_one_partial_mask"),
    ],
)
def test_engram_gate_npu_precision_arbitrary_dim(t, hc, dim, mask_kind):
    pytest.importorskip("torch_npu")
    x, key, value, weight = _inputs(t, hc, dim, 20260920 + dim)
    image_mask = None
    if mask_kind == "partial":
        image_mask = torch.arange(t) % 2 == 1
    expected = _golden(x, key, value, weight, 1e-6, 1e-6, image_mask)
    actual = engram_gate(
        x.npu(),
        key.npu(),
        value.npu(),
        weight.npu(),
        None if image_mask is None else image_mask.npu(),
    )
    torch.npu.synchronize()
    actual_cpu = actual.cpu()
    max_error = (actual_cpu.float() - expected.float()).abs().max().item()
    assert torch.allclose(actual_cpu.float(), expected.float(), atol=2e-2, rtol=2e-2), (
        f"max absolute error: {max_error}"
    )
    if image_mask is not None:
        assert torch.equal(actual_cpu[image_mask], x[image_mask])


@pytest.mark.npu
def test_vector_core_count_query():
    pytest.importorskip("torch_npu")
    count = _module._vector_core_count()
    assert count > 0
    assert _module._vector_core_count() == count


@pytest.mark.npu
@pytest.mark.parametrize(
    "t,hc,dim,forced_blocks",
    [
        pytest.param(72, 4, 5120, 8, id="reduced_blocks_specialized"),
        pytest.param(61, 3, 5120, 7, id="reduced_blocks_generic"),
        pytest.param(5, 4, 5120, 3, id="blocks_below_hc"),
    ],
)
def test_engram_gate_forced_block_count_precision(
    t, hc, dim, forced_blocks, monkeypatch
):
    pytest.importorskip("torch_npu")
    x, key, value, weight = _inputs(t, hc, dim, 20260924 + t)
    image_mask = torch.arange(t) % 2 == 1
    expected = _golden(x, key, value, weight, 1e-6, 1e-6, image_mask)
    monkeypatch.setattr(_module, "_vector_core_count", lambda: forced_blocks)
    actual = engram_gate(
        x.npu(), key.npu(), value.npu(), weight.npu(), image_mask.npu()
    )
    torch.npu.synchronize()
    actual_cpu = actual.cpu()
    max_error = (actual_cpu.float() - expected.float()).abs().max().item()
    assert torch.allclose(actual_cpu.float(), expected.float(), atol=2e-2, rtol=2e-2), (
        f"max absolute error: {max_error}"
    )
    assert torch.equal(actual_cpu[image_mask], x[image_mask])
