# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Torch dispatcher entry for FlashGDN backward."""

from functools import cache
from importlib import import_module
from pathlib import Path

import torch
from cann_ops_transformer.op_builder import OpBuilder, get_as_library
from torch.library import impl


@cache
def _kernel_module(name):
    prefix = ".op_dsl" if (Path(__file__).parent / "op_dsl").is_dir() else "..op_dsl"
    return import_module(f"{prefix}.{name}", __package__)


class FlashGdnGradOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("flash_gdn_grad", category="attention")

    def sources(self):
        return []

    def schema(self):
        return [
            (
                "flash_gdn_grad(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, "
                "Tensor do, Tensor dht, Tensor? Aqk=None, Tensor Akk, "
                "Tensor? metadata=None, *, Tensor? initial_state=None, "
                "Tensor? chunk_states=None, Tensor? w=None, Tensor? u=None, "
                "Tensor? v_new=None, Tensor? qg=None, Tensor? kg=None, "
                "Tensor? g_cumsum=None, Tensor? q_rstd=None, Tensor? k_rstd=None, "
                "Tensor? beta_sigmoid=None, Tensor? A_log=None, Tensor? dt_bias=None, "
                "Tensor? cu_seqlens=None, Tensor? seqused=None, float? scale=None, "
                "bool use_qk_l2norm_in_kernel=True, bool use_gate_in_kernel=True, "
                "bool use_beta_sigmoid_in_kernel=True, bool allow_neg_eigval=False, "
                'bool disable_recompute=False, str strategy="staged", '
                'str layout="TND") -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, '
                "-> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor?, Tensor?)"
            ),
        ]

    def register_meta(self):
        impl(get_as_library(), self.name, "Meta")(_flash_gdn_grad_meta)


def _flash_gdn_grad_meta(
    q,
    k,
    v,
    g,
    beta,
    do,
    dht,
    Aqk=None,
    Akk=None,
    metadata=None,
    *,
    initial_state=None,
    chunk_states=None,
    w=None,
    u=None,
    v_new=None,
    qg=None,
    kg=None,
    g_cumsum=None,
    q_rstd=None,
    k_rstd=None,
    beta_sigmoid=None,
    A_log=None,
    dt_bias=None,
    cu_seqlens=None,
    seqused=None,
    scale=None,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,
    allow_neg_eigval=False,
    disable_recompute=False,
    strategy="staged",
    layout="TND",
):
    torch._check(
        layout in ("BSND", "BNSD", "TND"), lambda: "unsupported FlashGDN layout"
    )
    torch._check(
        strategy in ("staged", "persistent"),
        lambda: "unsupported FlashGDN strategy",
    )
    # Akk 必选；Aqk 可选（省略时由反向重算）
    torch._check(Akk is not None and Akk.dim() == 4, lambda: "Akk must be [R, C, C]")
    if Aqk is not None:
        torch._check(Aqk.dim() == 4, lambda: "Aqk must be [R, C, C]")
    if disable_recompute:
        torch._check(
            v_new is not None
            and chunk_states is not None
            and beta_sigmoid is not None
            and qg is not None
            and kg is not None
            and g_cumsum is not None
            and q_rstd is not None
            and k_rstd is not None,
            lambda: "disable_recompute needs v_new/chunk_states/beta_sigmoid/"
            "qg/kg/g_cumsum/q_rstd/k_rstd",
        )
        torch._check(v_new.dim() == 3, lambda: "v_new must be [R, C, Dv]")
        torch._check(
            chunk_states.dim() == 3, lambda: "chunk_states must be [R, Dk, Dv]"
        )
    rank = 3 if layout == "TND" else 4
    torch._check(
        q.ndim == rank and k.ndim == rank and v.ndim == rank,
        lambda: "q, k and v ranks must match layout",
    )
    return (
        torch.empty_like(q, memory_format=torch.contiguous_format),
        torch.empty_like(k, memory_format=torch.contiguous_format),
        torch.empty_like(v, memory_format=torch.contiguous_format),
        torch.empty_like(
            dht, dtype=torch.float32, memory_format=torch.contiguous_format
        ),
        torch.empty_like(g, memory_format=torch.contiguous_format),
        torch.empty_like(beta, memory_format=torch.contiguous_format),
        None if A_log is None else torch.empty_like(A_log),
        None if dt_bias is None else torch.empty_like(dt_bias),
    )


@impl(get_as_library(), "flash_gdn_grad", "PrivateUse1")
def _flash_gdn_grad_npu(
    q,
    k,
    v,
    g,
    beta,
    do,
    dht,
    Aqk=None,
    Akk=None,
    metadata=None,
    *,
    initial_state=None,
    chunk_states=None,
    w=None,
    u=None,
    v_new=None,
    qg=None,
    kg=None,
    g_cumsum=None,
    q_rstd=None,
    k_rstd=None,
    beta_sigmoid=None,
    A_log=None,
    dt_bias=None,
    cu_seqlens=None,
    seqused=None,
    scale=None,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,
    allow_neg_eigval=False,
    disable_recompute=False,
    strategy="staged",
    layout="TND",
):
    """Return the eight FlashGDN gradients (dispatcher backend)."""
    return _run(
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
        initial_state,
        chunk_states,
        w,
        u,
        v_new,
        qg,
        kg,
        g_cumsum,
        q_rstd,
        k_rstd,
        beta_sigmoid,
        A_log,
        dt_bias,
        cu_seqlens,
        seqused,
        scale,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        disable_recompute,
        strategy,
        layout,
    )

    """Return the eight FlashGDN gradients in operator output order."""
    # 反向一律走 Python 入口：Aqk/Akk 必传，且需要 metadata / 缓存张量解析
    return _kernel_module("flash_gdn_grad").flash_gdn_grad_npu(
        q,
        k,
        v,
        g,
        beta,
        do,
        dht,
        Aqk,
        Akk,
        metadata=metadata,
        initial_state=initial_state,
        chunk_states=chunk_states,
        w=w,
        u=u,
        v_new=v_new,
        qg=qg,
        kg=kg,
        g_cumsum=g_cumsum,
        q_rstd=q_rstd,
        k_rstd=k_rstd,
        beta_sigmoid=beta_sigmoid,
        A_log=A_log,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        disable_recompute=disable_recompute,
        layout=layout,
    )


def _run(
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
    initial_state,
    chunk_states,
    w,
    u,
    v_new,
    qg,
    kg,
    g_cumsum,
    q_rstd,
    k_rstd,
    beta_sigmoid,
    A_log,
    dt_bias,
    cu_seqlens,
    seqused,
    scale,
    use_qk_l2norm_in_kernel,
    use_gate_in_kernel,
    use_beta_sigmoid_in_kernel,
    allow_neg_eigval,
    disable_recompute,
    strategy,
    layout,
):
    """反向一律走 Python 实现入口（需要 metadata 与缓存张量解析）。"""
    return _kernel_module("flash_gdn_grad").flash_gdn_grad_npu(
        q,
        k,
        v,
        g,
        beta,
        do,
        dht,
        Aqk=Aqk,
        Akk=Akk,
        metadata=metadata,
        initial_state=initial_state,
        chunk_states=chunk_states,
        w=w,
        u=u,
        v_new=v_new,
        qg=qg,
        kg=kg,
        g_cumsum=g_cumsum,
        q_rstd=q_rstd,
        k_rstd=k_rstd,
        beta_sigmoid=beta_sigmoid,
        A_log=A_log,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        scale=scale,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_gate_in_kernel=use_gate_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
        allow_neg_eigval=allow_neg_eigval,
        disable_recompute=disable_recompute,
        strategy=strategy,
        layout=layout,
    )


def flash_gdn_grad(
    q,
    k,
    v,
    g,
    beta,
    do,
    dht,
    Aqk=None,
    Akk=None,
    metadata=None,
    *,
    initial_state=None,
    chunk_states=None,
    w=None,
    u=None,
    v_new=None,
    qg=None,
    kg=None,
    g_cumsum=None,
    q_rstd=None,
    k_rstd=None,
    beta_sigmoid=None,
    A_log=None,
    dt_bias=None,
    cu_seqlens=None,
    seqused=None,
    scale=None,
    use_qk_l2norm_in_kernel=True,
    use_gate_in_kernel=True,
    use_beta_sigmoid_in_kernel=True,
    allow_neg_eigval=False,
    disable_recompute=False,
    layout="TND",
):
    """FlashGDN backward: returns dq, dk, dv, dh0, dg, dbeta, dA_log, dBias."""
    return _run(
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
        initial_state,
        chunk_states,
        w,
        u,
        v_new,
        qg,
        kg,
        g_cumsum,
        q_rstd,
        k_rstd,
        beta_sigmoid,
        A_log,
        dt_bias,
        cu_seqlens,
        seqused,
        scale,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        allow_neg_eigval,
        disable_recompute,
        strategy,
        layout,
    )


def flash_gdn_grad_metadata(q, *, layout="TND", cu_seqlens=None, seqused=None):
    """Build a reusable schedule for the Python adapter."""
    return _kernel_module("flash_gdn_grad_metadata").flash_gdn_grad_metadata(
        q,
        layout=layout,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
    )
