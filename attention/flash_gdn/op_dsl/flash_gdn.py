# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""GDN metadata, forward execution, and public routing entry points.

Scalar-gate chunk WU and fused state/output execute in flash_gdn_kernel.py.
"""

import math
import torch
from cannbotdsl.aot import export
from .flash_gdn_metadata import (
    HEAD_DIM,
    HEADER_SIZE,
    make_header,
    require,
    flash_gdn_metadata as _make_metadata,
    read_lengths,
    validate_length_specs,
)

from .flash_gdn_kernel import compile_kernel, run_device

FWD_OUTPUT_NAMES = (
    "out",
    "final_state",
    "chunk_states",
    "Akk",
    "Aqk",
    "w",
    "u",
    "qg",
    "kg",
    "v_new",
    "g_cumsum",
    "q_norm",
    "q_rstd",
    "k_norm",
    "k_rstd",
    "beta_sigmoid",
)


def _spec(
    q,
    k,
    v,
    g,
    beta,
    metadata,
    initial_state,
    cu_seqlens,
    seqused,
    scale,
    output_final_state,
    return_intermediate_states,
    use_qk_l2norm_in_kernel,
    use_gate_in_kernel,
    use_beta_sigmoid_in_kernel,
    allow_neg_eigval,
    disable_recompute,
    layout,
):
    require(layout in ("TND", "BSND", "BNSD"), "layout must be TND, BSND, or BNSD")
    flags = (
        output_final_state,
        return_intermediate_states,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        disable_recompute,
    )
    require(all(isinstance(f, bool) for f in flags), "forward flags must be bool")
    for name, t in (("q", q), ("k", k), ("v", v)):
        require(
            isinstance(t, torch.Tensor) and t.dtype == torch.bfloat16,
            f"{name} must be BF16",
        )
        require(
            t.is_contiguous() and t.device == q.device, f"{name} device/layout mismatch"
        )
    require(q.shape == k.shape, "q and k must have the same shape")
    rank = 3 if layout == "TND" else 4
    require(q.ndim == v.ndim == rank, "input rank does not match layout")
    for name, t in (("cu_seqlens", cu_seqlens), ("seqused", seqused)):
        if t is not None:
            require(
                isinstance(t, torch.Tensor)
                and t.dtype == torch.int32
                and t.ndim == 1
                and t.is_contiguous(),
                f"{name} must be a contiguous INT32 vector",
            )
    if layout == "TND":
        length, hq, dk = q.shape
        batch = (
            cu_seqlens.numel() - 1
            if cu_seqlens is not None
            else (seqused.numel() if seqused is not None else 1)
        )
        hv = v.shape[1]
        expected = (length, hv, HEAD_DIM)
        scalar = (length, hv)
    elif layout == "BSND":
        batch, length, hq, dk = q.shape
        hv = v.shape[2]
        expected = (batch, length, hv, HEAD_DIM)
        scalar = (batch, length, hv)
    else:
        batch, hq, length, dk = q.shape
        hv = v.shape[1]
        expected = (batch, hv, length, HEAD_DIM)
        scalar = (batch, hv, length)
    validate_length_specs(cu_seqlens, seqused, batch)
    header = make_header(batch, hq, hv, length, dk, v.shape[-1], layout)
    require(tuple(v.shape) == expected, "v shape mismatch")
    for name, t in (("g", g), ("beta", beta)):
        require(
            t.dtype in (torch.float32, torch.bfloat16) and tuple(t.shape) == scalar,
            f"{name} must have one scalar per token/value head",
        )
        require(
            t.device == q.device and t.is_contiguous(), f"{name} device/layout mismatch"
        )
    require(
        isinstance(metadata, torch.Tensor)
        and metadata.dtype == torch.int32
        and metadata.ndim == 1
        and metadata.numel() == HEADER_SIZE + batch
        and metadata.is_contiguous(),
        "metadata must be an int32 header from flash_gdn_metadata",
    )
    if initial_state is not None:
        require(
            initial_state.dtype == torch.float32
            and initial_state.device == q.device
            and tuple(initial_state.shape) == (batch, hv, HEAD_DIM, HEAD_DIM)
            and initial_state.is_contiguous(),
            "initial_state must be FP32 [B,Nv,Dv,Dk]",
        )
    scale = HEAD_DIM**-0.5 if scale is None else scale
    require(
        isinstance(scale, (float, int))
        and not isinstance(scale, bool)
        and math.isfinite(scale),
        "scale must be finite",
    )
    if q.device.type != "meta":
        require(
            metadata.device == q.device,
            "metadata must be on the q device, produced by AICPU",
        )
        lengths = read_lengths(cu_seqlens, seqused, batch, length, layout)
        require(
            metadata.detach().cpu().tolist() == header + lengths,
            "metadata ABI or dimensions/lengths do not match call",
        )
    return batch, hv, length, float(scale)


def flash_gdn_fwd(
    q,
    k,
    v,
    g,
    beta,
    metadata,
    initial_state=None,
    A_log=None,
    dt_bias=None,
    cu_seqlens=None,
    seqused=None,
    *,
    scale=None,
    output_final_state=False,
    return_intermediate_states=False,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    disable_recompute=True,
    layout="TND",
):
    spec = _spec(
        q,
        k,
        v,
        g,
        beta,
        metadata,
        initial_state,
        cu_seqlens,
        seqused,
        scale,
        output_final_state,
        return_intermediate_states,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        disable_recompute,
        layout,
    )
    batch, heads, length, scale = spec
    if use_gate_in_kernel:
        require(A_log is not None, "A_log is required for fused gate")
    for name, t in (("A_log", A_log), ("dt_bias", dt_bias)):
        if use_gate_in_kernel and t is not None:
            require(
                t.dtype == torch.float32
                and t.device == q.device
                and tuple(t.shape) == (heads,)
                and t.is_contiguous(),
                f"{name} must be FP32 [nheads_v]",
            )
    if q.device.type == "meta":
        out = torch.empty_like(v)
        final = (
            torch.empty(
                (batch, heads, HEAD_DIM, HEAD_DIM), device="meta", dtype=torch.float32
            )
            if output_final_state
            else None
        )
        chunks = None
        if return_intermediate_states:
            nc = (length + 63) // 64
            shape = (
                (nc + batch, heads, HEAD_DIM, HEAD_DIM)
                if layout == "TND"
                else (batch, nc, heads, HEAD_DIM, HEAD_DIM)
            )
            chunks = torch.empty(shape, dtype=torch.float32, device="meta")
        return (out, final, chunks) + (None,) * 13
    if q.device.type not in ("npu", "privateuseone"):
        raise NotImplementedError(
            "FlashGDN executes on Ascend NPU only; CPU is not an operator fallback"
        )

    out, final, chunks = run_device(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        scale,
        layout,
        metadata.detach().cpu().tolist(),
        A_log if use_gate_in_kernel else None,
        dt_bias if use_gate_in_kernel else None,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        return_intermediate_states,
        output_final_state,
    )
    return (out, final, chunks) + (None,) * 13


def flash_gdn(
    q,
    k,
    v,
    g,
    beta,
    metadata,
    initial_state=None,
    A_log=None,
    dt_bias=None,
    cu_seqlens=None,
    seqused=None,
    *,
    scale=None,
    output_final_state=False,
    return_intermediate_states=False,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    disable_recompute=True,
    layout="TND",
):
    return flash_gdn_fwd(
        q,
        k,
        v,
        g,
        beta,
        metadata,
        initial_state,
        A_log,
        dt_bias,
        cu_seqlens,
        seqused,
        scale=scale,
        output_final_state=output_final_state,
        return_intermediate_states=return_intermediate_states,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        disable_recompute=disable_recompute,
        layout=layout,
    )[:3]


def flash_gdn_metadata(
    cu_seqlens=None,
    seqused=None,
    *,
    batch_size,
    nheads_qk,
    nheads_v,
    seqlens,
    headdim_qk=128,
    headdim_v=128,
    enable_grad=False,
    layout="TND",
):
    """Build the host metadata consumed by both forward entry points."""
    return _make_metadata(
        cu_seqlens,
        seqused,
        batch_size=batch_size,
        nheads_qk=nheads_qk,
        nheads_v=nheads_v,
        seqlens=seqlens,
        headdim_qk=headdim_qk,
        headdim_v=headdim_v,
        enable_grad=enable_grad,
        layout=layout,
    )


def _compile_native_variant():
    return compile_kernel()


def _native_variants():
    yield (), {}


@export("flash_gdn")
def export_flash_gdn():
    compiled = _compile_native_variant()
    compiled.close()


@export("flash_gdn_fwd")
def export_flash_gdn_fwd():
    compiled = _compile_native_variant()
    compiled.close()


export_flash_gdn.__cannbot_aot_variants__ = (
    "_compile_native_variant",
    _native_variants,
)
export_flash_gdn_fwd.__cannbot_aot_variants__ = (
    "_compile_native_variant",
    _native_variants,
)
