# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Installed Torch adapter contract; run after installing torch_extension."""

import importlib
import os

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode


@pytest.fixture(scope="module")
def extension():
    return importlib.import_module(
        os.environ.get("KDA_TORCH_PACKAGE", "cann_ops_transformer")
    )


@pytest.mark.parametrize("layout", ["TND", "BNSD", "BSND"])
@pytest.mark.parametrize(
    "gate_dtype,beta_dtype",
    [(torch.float32, torch.bfloat16), (torch.bfloat16, torch.float32)],
)
@pytest.mark.parametrize(
    "fused_gate,has_bias", [(True, True), (True, False), (False, False)]
)
def test_fake_backward_contract(
    extension, layout, gate_dtype, beta_dtype, fused_gate, has_bias
):
    q_shape = {"TND": (65, 2, 128), "BNSD": (2, 2, 65, 128), "BSND": (2, 65, 2, 128)}[
        layout
    ]
    v_shape = {"TND": (65, 4, 128), "BNSD": (2, 4, 65, 128), "BSND": (2, 65, 4, 128)}[
        layout
    ]
    with FakeTensorMode():
        q = torch.empty(q_shape, dtype=torch.bfloat16)
        k = torch.empty_like(q)
        v = torch.empty(v_shape, dtype=torch.bfloat16)
        g = torch.empty(v_shape, dtype=gate_dtype)
        beta = torch.empty(v_shape[:-1], dtype=beta_dtype)
        state = torch.empty((2, 4, 128, 128), dtype=torch.float32)
        matrices = torch.empty((4, 4, 64, 64), dtype=torch.bfloat16)
        metadata = torch.empty((64,), dtype=torch.int32)
        bias = torch.empty((4, 128), dtype=torch.float32) if has_bias else None
        out = extension.flash_kda_grad(
            q,
            k,
            v,
            g,
            beta,
            v,
            state,
            matrices,
            matrices,
            metadata,
            dt_bias=bias,
            use_gate_in_kernel=fused_gate,
            layout=layout,
        )
        assert [tuple(t.shape) for t in out[:6]] == [
            q_shape,
            q_shape,
            v_shape,
            tuple(state.shape),
            v_shape,
            v_shape[:-1],
        ]
        assert [t.dtype for t in out[:6]] == [
            q.dtype,
            k.dtype,
            v.dtype,
            torch.float32,
            gate_dtype,
            beta_dtype,
        ]
        expected_stride = {"TND": (1, 66), "BNSD": (264, 66, 1), "BSND": (264, 1, 66)}[
            layout
        ]
        assert out[5].stride() == expected_stride
        assert (out[6] is not None) == fused_gate
        assert (out[7] is not None) == (fused_gate and has_bias)
        if fused_gate:
            assert out[6].shape == (4,) and out[6].dtype == torch.float32
            assert out[6].stride() == (8,)


def test_metadata_fake_without_lengths(extension):
    with FakeTensorMode():
        metadata = extension.flash_kda_grad_metadata(
            batch_size=2, n_heads_qk=2, n_heads_v=4, seqlens=65, layout="BNSD"
        )
        assert metadata.shape == (64,)
        assert metadata.dtype == torch.int32


def test_metadata_meta_dispatch(extension):
    lengths = torch.empty((3,), dtype=torch.int32, device="meta")
    metadata = extension.flash_kda_grad_metadata(
        lengths, batch_size=2, n_heads_qk=2, n_heads_v=4, seqlens=65
    )
    assert metadata.device.type == "meta"
    assert metadata.shape == (64,)


def _npu_ready():
    try:
        import torch_npu  # noqa: F401
    except (ImportError, RuntimeError):
        return False
    try:
        return torch.npu.is_available()
    except (AttributeError, RuntimeError):
        return False


def test_schema_and_dispatch_registered(extension):
    """Both schemas are registered, and Meta plus PrivateUse1 kernels exist."""
    grad = torch.ops.cann_ops_transformer.flash_kda_grad.default
    meta = torch.ops.cann_ops_transformer.flash_kda_grad_metadata.default
    assert grad is not None and meta is not None
    for name in (
        "cann_ops_transformer::flash_kda_grad",
        "cann_ops_transformer::flash_kda_grad_metadata",
    ):
        table = torch._C._dispatch_dump_table(name)
        assert "Meta" in table, f"{name} has no Meta kernel"
    # The backward dispatches on PrivateUse1; the metadata op falls back to a catch-all.
    assert "PrivateUse1" in torch._C._dispatch_dump_table(
        "cann_ops_transformer::flash_kda_grad"
    )


@pytest.mark.skipif(not _npu_ready(), reason="NPU device required")
def test_installed_package_end_to_end(extension):
    """End-to-end call through the installed package: metadata, dispatch, launcher, kernel.

    Values are the TTK suite's business; this test pins the ABI that the Python
    wrappers promise, so a broken schema, dispatch key or meta inference fails here.
    """
    device = "npu:0"
    length, n_qk, n_v, dim = 8, 1, 2, 128
    cu_seqlens = torch.tensor([0, length], dtype=torch.int32, device=device)
    metadata = extension.flash_kda_grad_metadata(
        cu_seqlens,
        batch_size=1,
        n_heads_qk=n_qk,
        n_heads_v=n_v,
        seqlens=length,
        headdim_qk=dim,
        headdim_v=dim,
        layout="TND",
    )
    assert metadata.dtype == torch.int32
    assert metadata.shape == (64,)
    assert metadata.device.type == "npu"

    chunks = (length + 63) // 64
    q = torch.randn(length, n_qk, dim, dtype=torch.bfloat16, device=device) * 0.1
    k = torch.randn_like(q)
    v = torch.randn(length, n_v, dim, dtype=torch.bfloat16, device=device) * 0.1
    g = torch.randn_like(v) * 0.1
    beta = torch.randn(length, n_v, dtype=torch.bfloat16, device=device) * 0.1
    do = torch.randn_like(v) * 0.1
    dht = torch.zeros(1, n_v, dim, dim, dtype=torch.float32, device=device)
    initial_state = torch.zeros(1, n_v, dim, dim, dtype=torch.float32, device=device)
    Aqk = torch.zeros(chunks, n_v, 64, 64, dtype=torch.bfloat16, device=device)
    Akk = torch.zeros_like(Aqk)
    A_log = torch.full((n_v,), -3.0, dtype=torch.float32, device=device)
    dt_bias = torch.zeros(n_v, dim, dtype=torch.float32, device=device)

    outputs = extension.flash_kda_grad(
        q,
        k,
        v,
        g,
        beta,
        do,
        dht,
        Aqk,
        Akk,
        metadata,
        initial_state=initial_state,
        A_log=A_log,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        disable_recompute=False,
        lower_bound=-5.0,
        layout="TND",
    )
    dq, dk, dv, dh0, dg, dbeta, da, dbias = outputs
    assert tuple(dq.shape) == (length, n_qk, dim)
    assert tuple(dk.shape) == (length, n_qk, dim)
    assert tuple(dv.shape) == (length, n_v, dim)
    assert tuple(dh0.shape) == (1, n_v, dim, dim)
    assert tuple(dg.shape) == (length, n_v, dim)
    assert tuple(dbeta.shape) == (length, n_v)
    assert da.shape == (n_v,) and da.dtype == torch.float32
    assert tuple(dbias.shape) == (n_v, dim) and dbias.dtype == torch.float32
    for name, tensor in zip(
        ("dq", "dk", "dv", "dh0", "dg", "dbeta", "dA", "dBias"), outputs
    ):
        assert tensor is not None, name
        assert torch.isfinite(tensor.float()).all(), name
