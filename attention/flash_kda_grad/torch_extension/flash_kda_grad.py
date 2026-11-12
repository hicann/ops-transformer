# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Torch dispatcher adapter for the CANNBotDSL FlashKDA backward kernel."""

from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Optional

import torch
from cann_ops_transformer.op_builder import OpBuilder, get_as_library
from torch.library import impl


@lru_cache(None)
def _kernel_module(name):
    # Installed wheels carry op_dsl beside this adapter; source checkouts
    # keep it beside the torch_extension directory. Never import test helpers.
    prefix = ".op_dsl" if (Path(__file__).parent / "op_dsl").is_dir() else "..op_dsl"
    return import_module(f"{prefix}.{name}", __package__)


class FlashKdaGradOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("flash_kda_grad", category="attention")

    def sources(self):
        # This operator has a DSL host launcher rather than an ACLNN wrapper.
        return []

    def schema(self):
        return [
            (
                "flash_kda_grad(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, "
                "Tensor do, Tensor dht, Tensor Aqk, Tensor Akk, Tensor? metadata=None, *, "
                "Tensor? initial_state=None, Tensor? chunk_states=None, Tensor? w=None, "
                "Tensor? u=None, Tensor? v_new=None, Tensor? qg=None, Tensor? kg=None, "
                "Tensor? g_cumsum=None, Tensor? q_rtsd=None, Tensor? k_rstd=None, "
                "Tensor? beta_sigmoid=None, Tensor? A_log=None, Tensor? dt_bias=None, "
                "Tensor? cu_seqlens=None, Tensor? seqused=None, float? scale=None, "
                "bool use_qk_l2norm_in_kernel=True, bool use_gate_in_kernel=True, "
                "bool use_beta_sigmoid_in_kernel=True, bool allow_neg_eigval=False, "
                "bool disable_recompute=False, float? lower_bound=None, "
                'str layout="TND") -> (Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, '
                "Tensor?, Tensor?)"
            ),
            (
                "flash_kda_grad_metadata(Tensor? cu_seqlens=None, Tensor? seqused=None, *, "
                "int batch_size, int n_heads_qk, int n_heads_v, int seqlens, "
                "int headdim_qk=128, int headdim_v=128, str layout='TND') -> Tensor"
            ),
        ]

    def register_meta(self):
        impl(get_as_library(), self.name, "Meta")(_flash_kda_grad_meta)
        torch.library.register_fake("cann_ops_transformer::flash_kda_grad_metadata")(
            _metadata_meta
        )


def _flash_kda_grad_meta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    metadata: Optional[torch.Tensor] = None,
    *,
    initial_state: Optional[torch.Tensor] = None,
    chunk_states: Optional[torch.Tensor] = None,
    w: Optional[torch.Tensor] = None,
    u: Optional[torch.Tensor] = None,
    v_new: Optional[torch.Tensor] = None,
    qg: Optional[torch.Tensor] = None,
    kg: Optional[torch.Tensor] = None,
    g_cumsum: Optional[torch.Tensor] = None,
    q_rtsd: Optional[torch.Tensor] = None,
    k_rstd: Optional[torch.Tensor] = None,
    beta_sigmoid: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    disable_recompute: bool = False,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
):
    torch._check(
        layout in ("TND", "BNSD", "BSND"), lambda: "unsupported FlashKDA layout"
    )
    expected_rank = 3 if layout == "TND" else 4
    torch._check(q.dim() == expected_rank, lambda: "q rank does not match layout")
    torch._check(v.dim() == expected_rank, lambda: "v rank does not match layout")
    torch._check(q.shape[-1] == 128, lambda: "FlashKDA Grad requires Dqk=128")
    torch._check(v.shape[-1] == 128, lambda: "FlashKDA Grad requires Dv=128")
    heads = v.shape[1] if layout in ("TND", "BNSD") else v.shape[2]
    length = (
        v.shape[0]
        if layout == "TND"
        else v.shape[2]
        if layout == "BNSD"
        else v.shape[1]
    )
    padded_length = length + length % 2
    if layout == "TND":
        dbeta = torch.empty(
            (heads, padded_length), dtype=beta.dtype, device=beta.device
        )[:, :length].t()
    else:
        dbeta = torch.empty(
            (v.shape[0], heads, padded_length), dtype=beta.dtype, device=beta.device
        )[:, :, :length]
        if layout == "BSND":
            dbeta = dbeta.transpose(1, 2)
    da = (
        torch.empty((heads, 8), dtype=torch.float32, device=q.device)[:, 0]
        if use_gate_in_kernel
        else None
    )
    dbias = (
        torch.empty((heads, 128), dtype=torch.float32, device=q.device)
        if use_gate_in_kernel and dt_bias is not None
        else None
    )
    return (
        torch.empty_like(q),
        torch.empty_like(k),
        torch.empty_like(v),
        torch.empty(dht.shape, dtype=torch.float32, device=dht.device),
        torch.empty_like(g),
        dbeta,
        da,
        dbias,
    )


def _metadata_meta(
    cu_seqlens=None,
    seqused=None,
    *,
    batch_size,
    n_heads_qk,
    n_heads_v,
    seqlens,
    headdim_qk=128,
    headdim_v=128,
    layout="TND",
):
    torch._check(batch_size > 0, lambda: "batch_size must be positive")
    torch._check(n_heads_qk > 0, lambda: "n_heads_qk must be positive")
    torch._check(n_heads_v > 0, lambda: "n_heads_v must be positive")
    torch._check(
        n_heads_v % n_heads_qk == 0, lambda: "n_heads_v must be divisible by n_heads_qk"
    )
    torch._check(
        headdim_qk == 128 and headdim_v == 128,
        lambda: "FlashKDA requires head dimensions 128",
    )
    torch._check(
        layout in ("TND", "BNSD", "BSND"), lambda: "unsupported FlashKDA layout"
    )
    reference = cu_seqlens if cu_seqlens is not None else seqused
    device = reference.device if reference is not None else torch.device("npu")
    # Metadata ABI: eight header words and seven words per (sequence, head).
    return torch.empty(
        (8 + batch_size * n_heads_v * 7,), dtype=torch.int32, device=device
    )


_builder = FlashKdaGradOpBuilder()
_builder.ensure_initialized()


@impl(get_as_library(), "flash_kda_grad", "PrivateUse1")
def _flash_kda_grad(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    metadata: Optional[torch.Tensor] = None,
    *,
    initial_state: Optional[torch.Tensor] = None,
    chunk_states: Optional[torch.Tensor] = None,
    w: Optional[torch.Tensor] = None,
    u: Optional[torch.Tensor] = None,
    v_new: Optional[torch.Tensor] = None,
    qg: Optional[torch.Tensor] = None,
    kg: Optional[torch.Tensor] = None,
    g_cumsum: Optional[torch.Tensor] = None,
    q_rtsd: Optional[torch.Tensor] = None,
    k_rstd: Optional[torch.Tensor] = None,
    beta_sigmoid: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    disable_recompute: bool = False,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
):
    return _kernel_module("flash_kda_grad").flash_kda_grad_npu(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        do=do,
        dht=dht,
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
        q_rtsd=q_rtsd,
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
        lower_bound=lower_bound,
        layout=layout,
    )


@torch.library.register_kernel("cann_ops_transformer::flash_kda_grad_metadata", None)
def _metadata(
    cu_seqlens=None,
    seqused=None,
    *,
    batch_size,
    n_heads_qk,
    n_heads_v,
    seqlens,
    headdim_qk=128,
    headdim_v=128,
    layout="TND",
):
    return _kernel_module("flash_kda_grad_metadata").flash_kda_grad_metadata(
        cu_seqlens,
        seqused,
        batch_size=batch_size,
        n_heads_qk=n_heads_qk,
        n_heads_v=n_heads_v,
        seqlens=seqlens,
        headdim_qk=headdim_qk,
        headdim_v=headdim_v,
        layout=layout,
    )


def flash_kda_grad(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    metadata: Optional[torch.Tensor] = None,
    *,
    initial_state: Optional[torch.Tensor] = None,
    chunk_states: Optional[torch.Tensor] = None,
    w: Optional[torch.Tensor] = None,
    u: Optional[torch.Tensor] = None,
    v_new: Optional[torch.Tensor] = None,
    qg: Optional[torch.Tensor] = None,
    kg: Optional[torch.Tensor] = None,
    g_cumsum: Optional[torch.Tensor] = None,
    q_rtsd: Optional[torch.Tensor] = None,
    k_rstd: Optional[torch.Tensor] = None,
    beta_sigmoid: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    scale: Optional[float] = None,
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    allow_neg_eigval: bool = False,
    disable_recompute: bool = False,
    lower_bound: Optional[float] = None,
    layout: str = "TND",
):
    """Return dq, dk, dv, dh0, dg, dbeta, dA, dBias using saved caches or recompute."""
    return torch.ops.cann_ops_transformer.flash_kda_grad(
        q=q,
        k=k,
        v=v,
        g=g,
        beta=beta,
        do=do,
        dht=dht,
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
        q_rtsd=q_rtsd,
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
        lower_bound=lower_bound,
        layout=layout,
    )


def flash_kda_grad_metadata(
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    *,
    batch_size: int,
    n_heads_qk: int,
    n_heads_v: int,
    seqlens: int,
    headdim_qk: int = 128,
    headdim_v: int = 128,
    layout: str = "TND",
) -> torch.Tensor:
    """Prepare the backward metadata snapshot before invoking the backward op."""
    return torch.ops.cann_ops_transformer.flash_kda_grad_metadata(
        cu_seqlens,
        seqused,
        batch_size=batch_size,
        n_heads_qk=n_heads_qk,
        n_heads_v=n_heads_v,
        seqlens=seqlens,
        headdim_qk=headdim_qk,
        headdim_v=headdim_v,
        layout=layout,
    )
