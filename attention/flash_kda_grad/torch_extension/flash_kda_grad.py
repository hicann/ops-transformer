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
                "Tensor? g_cumsum=None, Tensor? q_rstd=None, Tensor? k_rstd=None, "
                "Tensor? beta_sigmoid=None, Tensor? A_log=None, Tensor? dt_bias=None, "
                "Tensor? cu_seqlens=None, int[]? cu_seqlens_cpu=None, Tensor? seqused=None, float? scale=None, "
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
    q_rstd: Optional[torch.Tensor] = None,
    k_rstd: Optional[torch.Tensor] = None,
    beta_sigmoid: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_cpu: Optional[list[int]] = None,
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
    if cu_seqlens_cpu is not None:
        raise ValueError("cu_seqlens_cpu must be None")
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
    expected_beta = (length, heads) if layout == "TND" else (v.shape[0], length, heads)
    torch._check(
        beta.shape == expected_beta, lambda: "beta shape does not match layout"
    )
    torch._check(
        g.shape == (*expected_beta, 128), lambda: "g shape does not match layout"
    )
    torch._check(k.shape == q.shape, lambda: "k must match q")
    torch._check(do.shape == (*expected_beta, 128), lambda: "do must use BSND or TND")
    batch = dht.shape[0] if layout == "TND" else v.shape[0]
    state_shape = (batch, heads, 128, 128)
    torch._check(dht.shape == state_shape, lambda: "dht shape does not match layout")
    chunks = (length + 63) // 64
    prefix = (chunks + batch, heads) if layout == "TND" else (batch, chunks, heads)
    q_heads = q.shape[2 if layout == "BSND" else 1]
    cache_prefix = (heads, length) if layout == "TND" else (batch, heads, length)
    norm_shape = (q_heads, length) if layout == "TND" else (batch, q_heads, length)
    for name, tensor, shape in (
        ("Aqk", Aqk, (*cache_prefix, 64)),
        ("Akk", Akk, (*cache_prefix, 64)),
        ("initial_state", initial_state, state_shape),
        ("chunk_states", chunk_states, (*prefix, 128, 128)),
        ("w", w, (*cache_prefix, 128)),
        ("u", u, (*cache_prefix, 128)),
        ("qg", qg, (*cache_prefix, 128)),
        ("kg", kg, (*cache_prefix, 128)),
        ("v_new", v_new, (*cache_prefix, 128)),
        ("g_cumsum", g_cumsum, (*cache_prefix, 128)),
        ("q_rstd", q_rstd, norm_shape),
        ("k_rstd", k_rstd, norm_shape),
        ("beta_sigmoid", beta_sigmoid, cache_prefix),
    ):
        if tensor is not None:
            torch._check(
                tensor.shape == shape, lambda: f"{name} must have shape {shape}"
            )
    if chunk_states is not None:
        torch._check(
            chunk_states.dtype == torch.bfloat16, lambda: "chunk_states must be BF16"
        )
    dbeta = torch.empty_like(beta)
    da = (
        torch.empty((heads, 8), dtype=torch.float32, device=q.device)[:, 0]
        if use_gate_in_kernel
        else None
    )
    if dt_bias is not None:
        torch._check(
            dt_bias.shape in ((heads, 128), (heads * 128,)),
            lambda: "invalid dt_bias shape",
        )
    dbias = (
        torch.empty_like(dt_bias, dtype=torch.float32, device=q.device)
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
    q_rstd: Optional[torch.Tensor] = None,
    k_rstd: Optional[torch.Tensor] = None,
    beta_sigmoid: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_cpu: Optional[list[int]] = None,
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
        q_rstd=q_rstd,
        k_rstd=k_rstd,
        beta_sigmoid=beta_sigmoid,
        A_log=A_log,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
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
    q_rstd: Optional[torch.Tensor] = None,
    k_rstd: Optional[torch.Tensor] = None,
    beta_sigmoid: Optional[torch.Tensor] = None,
    A_log: Optional[torch.Tensor] = None,
    dt_bias: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_cpu: Optional[list[int]] = None,
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
        q_rstd=q_rstd,
        k_rstd=k_rstd,
        beta_sigmoid=beta_sigmoid,
        A_log=A_log,
        dt_bias=dt_bias,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
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
