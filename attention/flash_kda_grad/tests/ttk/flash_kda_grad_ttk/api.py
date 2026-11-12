# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""TTK entry for ``flash_kda_grad`` — one callable, two device paths.

``npu`` is what the TTK E2E case points at (``api_name=flash_kda_grad_ttk.npu``):

* on NPU tensors it builds the forward cache, generates the AICPU metadata and
  runs the real ``flash_kda_grad_npu`` kernel;
* on host tensors (``ttk e2e --backend cpu``, or a machine without a device) it
  replays the same math with the BF16-emulating staged reference.

Both paths consume the **same** forward cache: the cache is built once from the
case inputs and then fed, bit-identically, to the device kernel and to the
``third_party`` reference. That is deliberate — a backward-operator acceptance
test must not fold forward-kernel error into the backward comparison, otherwise
a failure cannot be attributed.

Inputs follow the operator contract (BNSD shown; BSND swaps the S/N axes):

===============  ==========================  ==========================================
q, k             [B, Nq, S, D] BF16           raw query / key (L2-normalized in kernel)
v                [B, Nv, S, D] BF16           value
g                [B, Nv, S, D] BF16           raw gate input
beta             [B, Nv, S] BF16              raw write logits
A_log            [Nv] FP32                    log time scale (fused gate)
dt_bias          [Nv, D] FP32                 per-channel bias (fused gate)
initial_state    [B, Nv, Dv, Dk] FP32         chunk-0 recurrent state
do               [B, Nv, S, D] BF16           output upstream gradient
dht              [B, Nv, Dv, Dk] FP32         final-state upstream gradient
===============  ==========================  ==========================================

Returns ``(dq, dk, dv, dh0, dg, dbeta, dA, dBias)`` with the operator's dtypes:
``dq/dk/dv/dg/dbeta`` BF16 (input dtype), ``dh0/dA/dBias`` FP32.
"""

import os
import sys
from pathlib import Path
from typing import Optional

import torch

from . import golden

__all__ = ["npu"]


_OPERATOR_ROOT = Path(__file__).resolve().parents[3]
_OPERATOR_CACHE: dict[str, object] = {}


def _source_loader():
    """Load only this checkout; a broken source import must fail the test."""
    test_dir = str(_OPERATOR_ROOT / "tests")
    added = test_dir not in sys.path
    if added:
        sys.path.insert(0, test_dir)
    try:
        from _kdag_paths import load_sample  # noqa: PLC0415
    finally:
        if added:
            sys.path.remove(test_dir)
    return load_sample


def _load_operator():
    """Return the backward entry and metadata from the adjacent op_dsl directory."""
    if "pair" not in _OPERATOR_CACHE:
        load_sample = _source_loader()
        integration = load_sample("flash_kda_grad/flash_kda_grad.py")
        metadata = load_sample("flash_kda_grad/flash_kda_grad_metadata.py")
        _OPERATOR_CACHE["pair"] = (
            integration.flash_kda_grad_npu,
            metadata.flash_kda_grad_metadata,
        )
    return _OPERATOR_CACHE["pair"]


def _resolve_lower_bound(gate_mode: str, lower_bound: float) -> Optional[float]:
    mode = str(gate_mode).lower()
    if mode == "softplus":
        return None
    if mode != "bounded":
        raise ValueError(
            f"gate_mode must be 'bounded' or 'softplus', got {gate_mode!r}"
        )
    return float(lower_bound)


def _token_major(t: torch.Tensor, layout: str) -> torch.Tensor:
    return golden.token_major(t, layout)


# ---------------------------------------------------------------------------
# Cache construction
# ---------------------------------------------------------------------------


def _build_cache(
    q, k, v, g, beta, initial_state, A_log, dt_bias, *, use_fwd: bool, **kwargs
):
    """Forward cache for the backward operator.

    ``use_fwd=False`` (default) evaluates the reference chunkwise forward on the
    host and then uploads the cache. The upload cost is bounded by the case
    size and buys bit-identical caches for the device and the peer.
    """
    if use_fwd:
        device = q.device
        fwd_pair = _load_forward()
        return fwd_pair(
            q,
            k,
            v,
            g,
            beta,
            initial_state,
            A_log,
            dt_bias,
            scale=kwargs["scale"],
            gate_mode=kwargs["gate_mode"],
            lower_bound=kwargs["lower_bound"],
            allow_neg_eigval=kwargs["allow_neg_eigval"],
            layout=kwargs["layout"],
            use_qk_l2norm_in_kernel=kwargs["use_qk_l2norm_in_kernel"],
            use_gate_in_kernel=kwargs["use_gate_in_kernel"],
            use_beta_sigmoid_in_kernel=kwargs["use_beta_sigmoid_in_kernel"],
            device=device,
            cu_seqlens=kwargs.get("cu_seqlens"),
            seqused=kwargs.get("seqused"),
        )

    host = {
        name: None if tensor is None else tensor.detach().to("cpu")
        for name, tensor in (
            ("q", q),
            ("k", k),
            ("v", v),
            ("g", g),
            ("beta", beta),
            ("initial_state", initial_state),
            ("A_log", A_log),
            ("dt_bias", dt_bias),
        )
    }
    kwargs = dict(kwargs)
    for name in ("cu_seqlens", "seqused"):
        if kwargs.get(name) is not None:
            kwargs[name] = kwargs[name].detach().cpu()
    cache = golden.build_cache(
        host["q"],
        host["k"],
        host["v"],
        host["g"],
        host["beta"],
        host["initial_state"],
        host["A_log"],
        host["dt_bias"],
        emulate_bf16=True,
        **kwargs,
    )
    return _to_device(cache, q.device)


def _to_device(cache: "golden.Cache", device: torch.device) -> "golden.Cache":
    moved = {}
    for field_name, value in cache.__dict__.items():
        moved[field_name] = None if value is None else value.to(device)
    return golden.Cache(**moved)


def _load_forward():
    """Optional full-pipeline mode: run ``flash_kda_fwd`` on the device.

    Kept lazy so the default path never imports the forward operator.
    """
    load_sample = _source_loader()

    fwd = load_sample("flash_kda_fwd/flash_kda_fwd.py")
    fwd_meta = load_sample("flash_kda_fwd/flash_kda_fwd_metadata.py")

    def run(
        q,
        k,
        v,
        g,
        beta,
        initial_state,
        A_log,
        dt_bias,
        *,
        scale,
        gate_mode,
        lower_bound,
        allow_neg_eigval,
        layout,
        use_qk_l2norm_in_kernel,
        use_gate_in_kernel,
        use_beta_sigmoid_in_kernel,
        device,
        cu_seqlens=None,
        seqused=None,
    ):
        forward_lengths = cu_seqlens
        if seqused is not None:
            forward_lengths = torch.cat(
                (seqused.new_zeros(1), seqused.cumsum(0, dtype=torch.int32))
            )
        metadata = fwd_meta.flash_kda_fwd_metadata(
            q, v, initial_state, layout, forward_lengths
        )
        outputs = fwd.flash_kda_fwd(
            q,
            k,
            v,
            g,
            beta,
            float(scale),
            initial_state,
            A_log,
            dt_bias,
            _resolve_lower_bound(gate_mode, lower_bound),
            layout,
            metadata,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_gate_in_kernel=use_gate_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            disable_recompute=True,
            return_intermediate_state=True,
            allow_neg_eigval=allow_neg_eigval,
            output_final_state=True,
        )
        (
            out,
            final_state,
            q_norm,
            k_norm,
            q_rstd,
            k_rstd,
            beta_sigmoid,
            Aqk,
            Akk,
            w,
            u,
            qg,
            kg,
            v_new,
            g_cumsum,
            chunk_states,
        ) = outputs
        return golden.Cache(
            out=out,
            final_state=final_state,
            q_norm=q_norm if use_qk_l2norm_in_kernel else None,
            k_norm=k_norm if use_qk_l2norm_in_kernel else None,
            q_rstd=q_rstd if use_qk_l2norm_in_kernel else None,
            k_rstd=k_rstd if use_qk_l2norm_in_kernel else None,
            beta_sigmoid=beta_sigmoid,
            Aqk=Aqk,
            Akk=Akk,
            w=w,
            u=u,
            qg=qg,
            kg=kg,
            v_new=v_new,
            g_cumsum=g_cumsum,
            chunk_states=chunk_states,
            beta_effective=None,
        )

    return run


# ---------------------------------------------------------------------------
# Public entry
# ---------------------------------------------------------------------------


_RECOMPUTE = os.environ.get("KDA_GRAD_RECOMPUTE", "0") == "1"


def _operator_cache(
    cache, *, batch: int, storage: int, layout="BNSD", bounds=None, use_fwd=False
):
    """Reshape the flat token-major cache into the operator's BNSD cache ABI.

    The renamed operator validates ``(B, Nv, S, D)`` token caches,
    ``(B, Nv, chunks, 64, 64)`` Aqk/Akk and ``(B, Nv, chunks, Dk, Dv)`` chunk
    states, while the host reference keeps everything token-major
    (``(T, Nv, D)`` / ``(chunks_total, Nv, 64, 64)``). Both are views of the
    same values, so the reshape does not change any number.
    """

    def tokens(tensor, *, scalar=False):
        if tensor is None:
            return None
        if layout == "TND":
            return tensor
        if tensor.ndim == (3 if scalar else 4):
            return tensor
        if tensor.ndim == 2:
            return (
                tensor.reshape(batch, storage, tensor.shape[1])
                .permute(0, 2, 1)
                .contiguous()
            )
        return (
            tensor.reshape(batch, storage, tensor.shape[1], tensor.shape[2])
            .permute(0, 2, 1, 3)
            .contiguous()
        )

    def chunked(tensor):
        if tensor is None or use_fwd:
            return tensor
        if bounds is None:
            raise ValueError("reference cache packing requires sequence bounds")
        heads = tensor.shape[1]
        if layout == "TND":
            result = tensor.new_zeros(
                ((storage + 63) // 64 + batch, heads, *tensor.shape[2:])
            )
        else:
            result = tensor.new_zeros(
                (batch, heads, (storage + 63) // 64, *tensor.shape[2:])
            )
        cursor = 0
        for seq, (start, end) in enumerate(bounds):
            count = (end - start + 63) // 64
            if layout == "TND":
                slot = start // 64 + seq
                result[slot : slot + count] = tensor[cursor : cursor + count]
            else:
                result[seq, :, :count] = tensor[cursor : cursor + count].transpose(0, 1)
            cursor += count
        return result

    akk = chunked(cache.Akk)
    return {
        "Aqk": chunked(cache.Aqk),
        "Akk": akk,
        "chunk_states": chunked(cache.chunk_states),
        "qg": tokens(cache.qg),
        "kg": tokens(cache.kg),
        "w": tokens(cache.w),
        "u": tokens(cache.u),
        "v_new": tokens(cache.v_new),
        "g_cumsum": tokens(cache.g_cumsum),
        "q_rtsd": tokens(cache.q_rstd, scalar=True),
        "k_rstd": tokens(cache.k_rstd, scalar=True),
        "beta_sigmoid": tokens(cache.beta_sigmoid, scalar=True),
    }


def npu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    *,
    scale: Optional[float] = None,
    gate_mode: str = "bounded",
    lower_bound: float = -5.0,
    allow_neg_eigval: bool = False,
    layout: str = "BNSD",
    cache_source: str = "ref",
    use_qk_l2norm_in_kernel: bool = True,
    use_gate_in_kernel: bool = True,
    use_beta_sigmoid_in_kernel: bool = True,
    disable_recompute: Optional[bool] = None,
):
    """FlashKDA backward; device dispatch on ``q.device.type``.

    ``disable_recompute=False`` asks the NPU kernel to rebuild seven forward
    caches; True consumes saved caches. None preserves the legacy
    KDA_GRAD_RECOMPUTE environment setting (unset means saved caches).
    This switch does not change the CPU golden/peer arithmetic.
    """
    if q.dtype != k.dtype or q.shape != k.shape:
        raise ValueError("q and k must share shape and dtype")
    initial_state = golden.resolve_initial_state(
        q, v, initial_state, layout, cu_seqlens
    )
    bounds = golden.sequence_bounds(q, initial_state, layout, cu_seqlens, seqused)
    for lengths in (cu_seqlens, seqused):
        if lengths is not None and lengths.device != q.device:
            raise ValueError("sequence metadata must be on the same device as q")
    scale = float(scale) if scale is not None else float(q.shape[-1]) ** -0.5
    bound = _resolve_lower_bound(gate_mode, lower_bound)
    common = dict(
        scale=scale,
        gate_mode=str(gate_mode).lower(),
        lower_bound=None if lower_bound is None else float(lower_bound),
        allow_neg_eigval=bool(allow_neg_eigval),
        layout=layout,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        use_qk_l2norm_in_kernel=bool(use_qk_l2norm_in_kernel),
        use_gate_in_kernel=bool(use_gate_in_kernel),
        use_beta_sigmoid_in_kernel=bool(use_beta_sigmoid_in_kernel),
    )

    if q.device.type != "npu":
        # Host replay: identical signature and identical cache dtypes.
        cache = golden.build_cache(
            q, k, v, g, beta, initial_state, A_log, dt_bias, emulate_bf16=True, **common
        )
        return golden.staged_backward(
            q, k, v, g, beta, do, dht, cache, A_log=A_log, dt_bias=dt_bias, **common
        )

    bwd, bwd_metadata = _load_operator()
    use_fwd = str(cache_source).lower() == "fwd"
    cache = _build_cache(
        q, k, v, g, beta, initial_state, A_log, dt_bias, use_fwd=use_fwd, **common
    )
    # With L2Norm enabled, backward consumes saved normalized Q/K and the
    # inverse norms; it returns gradients of the original raw inputs.
    q_for_bwd, k_for_bwd = q, k
    if use_qk_l2norm_in_kernel:
        if use_fwd:
            # Device forward saves Q/K in BNSD, including for BSND inputs.
            q_for_bwd = cache.q_norm
            k_for_bwd = cache.k_norm
            if layout == "BSND":
                q_for_bwd = q_for_bwd.transpose(1, 2).contiguous()
                k_for_bwd = k_for_bwd.transpose(1, 2).contiguous()
        else:
            # The host reference saves flattened token-major Q/K.
            q_for_bwd = golden.restore_layout(cache.q_norm, q, layout)
            k_for_bwd = golden.restore_layout(cache.k_norm, k, layout)
    metadata = bwd_metadata(
        cu_seqlens,
        seqused,
        batch_size=int(initial_state.shape[0]),
        n_heads_qk=int(q.shape[-2] if layout == "BSND" else q.shape[1]),
        n_heads_v=int(v.shape[-2] if layout == "BSND" else v.shape[1]),
        seqlens=int(
            q.shape[0]
            if layout == "TND"
            else q.shape[1]
            if layout == "BSND"
            else q.shape[2]
        ),
        layout=layout,
    )
    op_cache = _operator_cache(
        cache,
        batch=int(initial_state.shape[0]),
        storage=int(
            q.shape[0]
            if layout == "TND"
            else q.shape[1]
            if layout == "BSND"
            else q.shape[2]
        ),
        layout=layout,
        bounds=bounds,
        use_fwd=use_fwd,
    )
    # Explicit per-case attributes take priority over the legacy environment.
    # In recompute mode retain Aqk/Akk, norm and beta caches, but pass None for
    # the seven tracks the kernel rebuilds from normalized Q/K and initial_state.
    recompute = _RECOMPUTE if disable_recompute is None else not bool(disable_recompute)
    return bwd(
        q=q_for_bwd,
        k=k_for_bwd,
        v=v,
        g=g,
        beta=beta,
        do=do,
        dht=dht,
        Aqk=op_cache["Aqk"],
        Akk=op_cache["Akk"],
        metadata=metadata,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        initial_state=None if not recompute else initial_state,
        chunk_states=None if recompute else op_cache["chunk_states"],
        w=None if recompute else op_cache["w"],
        u=None if recompute else op_cache["u"],
        v_new=None if recompute else op_cache["v_new"],
        qg=None if recompute else op_cache["qg"],
        kg=None if recompute else op_cache["kg"],
        g_cumsum=None if recompute else op_cache["g_cumsum"],
        q_rtsd=op_cache["q_rtsd"] if use_qk_l2norm_in_kernel else None,
        k_rstd=op_cache["k_rstd"] if use_qk_l2norm_in_kernel else None,
        beta_sigmoid=op_cache["beta_sigmoid"] if use_beta_sigmoid_in_kernel else None,
        A_log=A_log if use_gate_in_kernel else None,
        dt_bias=dt_bias if use_gate_in_kernel else None,
        scale=scale,
        use_qk_l2norm_in_kernel=bool(use_qk_l2norm_in_kernel),
        use_gate_in_kernel=bool(use_gate_in_kernel),
        use_beta_sigmoid_in_kernel=bool(use_beta_sigmoid_in_kernel),
        allow_neg_eigval=bool(allow_neg_eigval),
        lower_bound=bound,
        layout=layout,
        disable_recompute=not recompute,
    )
