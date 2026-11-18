# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Dispatcher/Meta/contract tests; NPU cases explicitly opt in."""

import os
import pytest
import torch

from cann_ops_transformer import flash_gdn, flash_gdn_fwd, flash_gdn_metadata

NPU = pytest.mark.skipif(
    os.environ.get("RUN_GDN_NPU") != "1",
    reason="set RUN_GDN_NPU=1 on a free Ascend device",
)


def metadata(length=3, heads=1, batch=1, layout="TND", device="cpu", **kwargs):
    if device == "npu":
        return flash_gdn_metadata(
            batch_size=batch,
            nheads_qk=heads,
            nheads_v=heads,
            seqlens=length,
            layout=layout,
            **kwargs,
        )
    # Host/Meta validation fixture only, never a backend for the metadata DUT.
    from cann_ops_transformer.ops.attention.flash_gdn.op_dsl.flash_gdn_metadata import (
        make_header,
        read_lengths,
    )

    cu, used = kwargs.get("cu_seqlens"), kwargs.get("seqused")
    words = make_header(batch, heads, heads, length, 128, 128, layout)
    return torch.tensor(
        words + read_lengths(cu, used, batch, length, layout),
        dtype=torch.int32,
        device="cpu",
    )


def inputs(layout="TND", length=3, heads=1, batch=1, device="meta"):
    if layout == "TND":
        shape, scalar = (length, heads, 128), (length, heads)
    elif layout == "BSND":
        shape, scalar = (batch, length, heads, 128), (batch, length, heads)
    else:
        shape, scalar = (batch, heads, length, 128), (batch, heads, length)
    q = torch.empty(shape, dtype=torch.bfloat16, device=device)
    return (
        q,
        torch.empty_like(q),
        torch.empty_like(q),
        torch.empty(scalar, dtype=torch.float32, device=device),
        torch.empty(scalar, dtype=torch.bfloat16, device=device),
    )


def test_public_dispatcher_entries():
    public = {
        "flash_gdn": flash_gdn,
        "flash_gdn_fwd": flash_gdn_fwd,
        "flash_gdn_metadata": flash_gdn_metadata,
    }
    for name, function in public.items():
        handle = getattr(torch.ops.cann_ops_transformer, name).default
        assert callable(function)
        assert getattr(function, "default", function) is handle
        assert handle._schema.name == "cann_ops_transformer::" + name
    text = str(torch.ops.cann_ops_transformer.flash_gdn.default._schema)
    assert "float? scale=None" in text
    assert "use_gate_in_kernel=False" in text
    assert "use_qk_l2norm_in_kernel=False" in text
    assert "lower_bound" not in text


@pytest.mark.npu
@NPU
def test_aicpu_metadata_without_tensor_dispatch():
    md = metadata(65, device="npu")
    assert md.device.type == "npu" and md.dtype == torch.int32 and md.shape == (13,)
    assert md.tolist()[2:8] == [1, 1, 1, 65, 128, 128]
    assert md.tolist()[10] == 2


@pytest.mark.parametrize(
    "layout,batch,heads", [("TND", 1, 2), ("BSND", 2, 2), ("BNSD", 2, 2)]
)
@pytest.mark.parametrize("final", [False, True])
def test_meta_shape_and_optional_outputs(layout, batch, heads, final):
    args = inputs(layout, heads=heads, batch=batch)
    result = flash_gdn(
        *args,
        metadata(heads=heads, batch=batch, layout=layout),
        layout=layout,
        output_final_state=final,
    )
    assert len(result) == 3 and result[0].shape == args[2].shape
    assert result[0].dtype == torch.bfloat16 and result[0].device.type == "meta"
    assert result[2] is None
    if final:
        assert (
            result[1].shape == (batch, heads, 128, 128)
            and result[1].dtype == torch.float32
        )
    else:
        assert result[1] is None


def test_fwd_fixed_tuple_csv_order():
    result = flash_gdn_fwd(*inputs(), metadata())
    assert len(result) == 16
    assert all(item is None for item in result[1:])


def test_no_silent_gradient_loss():
    args = list(inputs())
    args[0].requires_grad_()
    with pytest.raises(NotImplementedError, match="inference-only"):
        flash_gdn(*args, metadata())


@pytest.mark.parametrize("scale", [float("nan"), float("inf")])
def test_bad_scale(scale):
    with pytest.raises(ValueError, match="scale"):
        flash_gdn(*inputs(), metadata(), scale=scale)


def test_bad_gate_shape():
    q, k, v, g, beta = inputs()
    with pytest.raises(ValueError, match="one scalar"):
        flash_gdn(q, k, v, torch.empty_like(q), beta, metadata())


def test_cpu_is_not_runtime_golden_fallback():
    with pytest.raises(NotImplementedError, match="NPU only"):
        flash_gdn(*inputs(device="cpu"), metadata())


def golden(q, k, v, g, beta, initial, scale):
    # Testing only. Never imported by the runtime operator.
    state = initial.clone().double()
    out = []
    for i in range(q.shape[2]):
        state = state * g[:, :, i].double().exp()[..., None, None]
        prediction = (state * k[:, :, i].double()[..., None, :]).sum(-1)
        correction = beta[:, :, i].double()[..., None] * (
            v[:, :, i].double() - prediction
        )
        state = state + correction[..., :, None] * k[:, :, i].double()[..., None, :]
        out.append((state * q[:, :, i].double()[..., None, :]).sum(-1) * scale)
    return torch.stack(out, 2), state


@pytest.mark.npu
@NPU
@pytest.mark.parametrize(
    "layout,batch,heads,length",
    [
        ("TND", 1, 1, 1),
        ("TND", 1, 2, 3),
        ("BNSD", 2, 2, 8),
        ("BSND", 1, 2, 65),
        ("BNSD", 1, 1, 128),
    ],
)
@pytest.mark.parametrize("gate_dtype", [torch.float32, torch.bfloat16])
def test_npu_matches_independent_recurrence(layout, batch, heads, length, gate_dtype):
    import torch_npu  # noqa: F401

    torch.manual_seed(12)
    q = torch.nn.functional.normalize(
        torch.randn(batch, heads, length, 128), dim=-1
    ).to(torch.bfloat16)
    k = torch.nn.functional.normalize(
        torch.randn(batch, heads, length, 128), dim=-1
    ).to(torch.bfloat16)
    v = torch.randn_like(q)
    g = (-torch.rand(batch, heads, length) * 0.2).to(gate_dtype)
    beta = torch.rand(batch, heads, length).to(gate_dtype)
    initial = torch.randn(batch, heads, 128, 128) * 0.02
    ref_out, ref_state = golden(q, k, v, g, beta, initial, 0.7)

    def logical(x):
        if layout == "TND":
            return x.transpose(1, 2)[0].contiguous()
        if layout == "BSND":
            return x.transpose(1, 2).contiguous()
        return x.contiguous()

    values = [logical(x).npu() for x in (q, k, v, g, beta)]
    initial_npu = initial.npu()
    original = initial_npu.clone()
    with torch.inference_mode():
        result = flash_gdn_fwd(
            *values,
            metadata(length, heads, batch, layout, device="npu"),
            initial_state=initial_npu,
            layout=layout,
            output_final_state=True,
            scale=0.7,
        )
    torch.npu.synchronize()
    actual = result[0].cpu().float()
    torch.testing.assert_close(actual, logical(ref_out).float(), atol=0.01, rtol=0.01)
    torch.testing.assert_close(result[1].cpu(), ref_state.float(), atol=2e-5, rtol=2e-4)
    assert torch.equal(initial_npu, original)
    assert len(result) == 16 and all(item is None for item in result[2:])


@pytest.mark.npu
@NPU
def test_npu_simple_state_update_and_zero_scale():
    import torch_npu  # noqa: F401

    q = torch.zeros(1, 1, 2, 128, dtype=torch.bfloat16, device="npu")
    q[..., 0] = 1
    v = torch.zeros_like(q)
    v[:, :, 0, 0] = 2
    v[:, :, 1, 0] = 4
    g = torch.full((1, 1, 2), torch.tensor(0.5).log().item(), device="npu")
    beta = torch.full_like(g, 0.5)
    with torch.inference_mode():
        a = flash_gdn(
            q,
            q,
            v,
            g,
            beta,
            metadata(2, layout="BNSD", device="npu"),
            layout="BNSD",
            scale=1.0,
            output_final_state=True,
        )
        b = flash_gdn(
            q,
            q,
            v,
            g,
            beta,
            metadata(2, layout="BNSD", device="npu"),
            layout="BNSD",
            scale=0.0,
            output_final_state=True,
        )
    torch.npu.synchronize()
    torch.testing.assert_close(
        a[0][0, 0, :, 0].cpu().float(), torch.tensor([1.0, 2.25]), atol=1e-5, rtol=1e-5
    )
    assert torch.count_nonzero(b[0]) == 0
    torch.testing.assert_close(a[1], b[1], atol=0, rtol=0)


@pytest.mark.npu
@NPU
def test_metadata_stays_on_npu_with_meta_default_device():
    # Scalar-only calls have no Meta dispatch tensor; like KDA they target current NPU.
    with torch.device("meta"):
        md = flash_gdn_metadata(batch_size=1, nheads_qk=1, nheads_v=1, seqlens=3)
    assert md.device.type == "npu"


def test_registered_metadata_meta_dispatch():
    cu = torch.empty(3, dtype=torch.int32, device="meta")
    md = flash_gdn_metadata(cu, batch_size=2, nheads_qk=1, nheads_v=2, seqlens=9)
    assert md.device.type == "meta" and md.shape == (14,)


def test_meta_does_not_read_metadata_values():
    md = torch.empty(13, dtype=torch.int32, device="meta")
    assert flash_gdn(*inputs(), md)[0].device.type == "meta"


def test_disable_recompute_is_ignored_for_inference_outputs():
    assert all(
        x is None
        for x in flash_gdn_fwd(*inputs(), metadata(), disable_recompute=False)[1:]
    )


def test_corrupt_metadata_is_rejected_before_launch():
    md = metadata()
    md[0] = 0
    with pytest.raises(ValueError, match="metadata ABI"):
        flash_gdn(*inputs(device="cpu"), md)


def test_bad_metadata_dtype():
    with pytest.raises(ValueError, match="int32 header"):
        flash_gdn(*inputs(), metadata().to(torch.int64))


def test_qk_shape_mismatch():
    q, k, v, g, beta = inputs()
    with pytest.raises(ValueError, match="same shape"):
        flash_gdn(q, k[:2].contiguous(), v, g, beta, metadata())


def test_no_grad_allows_inference_meta_with_requires_grad_input():
    args = list(inputs())
    args[0].requires_grad_()
    with torch.no_grad():
        result = flash_gdn(*args, metadata())
    assert not result[0].requires_grad


def test_initial_state_shape_validation():
    state = torch.empty(1, 1, 64, 128, dtype=torch.float32, device="meta")
    with pytest.raises(ValueError, match="initial_state"):
        flash_gdn(*inputs(), metadata(), initial_state=state)
