#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Materialize the mixed_quant_flash_attn metadata slot before the main API call.

The native op is invoked directly by TTK with the tensors produced by the input
plugin.  When the CSV declares the metadata slot as dynamic (``tensor_view_shapes``
contains ``(-1,)``) TTK passes ``metadata=None`` and requires the ``npu_preprocess``
TestSpec hook (worker.py:814-825) to materialize it on device.  This hook builds a
real 1-D int32 metadata tensor via ``mixed_quant_flash_attn_metadata`` — the same
call the pytest ``core.npu.run_npu`` path uses — and returns it for the dynamic
slot so the ``metadata_checker`` (which demands rank 1) is satisfied.

Called after H2D, inside the device scope, before the main op.  Positional args
are the device-side input tensors in op-schema order; scalar attributes arrive as
keyword args.  Must NOT declare a ``context`` parameter.  Returns
``{"metadata": tensor}`` for the dynamic slot (or None when the slot is static and
filled in place).
"""

import torch


def get_attribute(kwargs, name, default=None):
    """Resolve an attribute from ``kwargs`` with an optional ``pytest_`` alias."""
    value = kwargs.get(name)
    if value is None:
        value = kwargs.get("pytest_%s" % name)
    return default if value is None else value


def _to_int_list(value):
    """Normalize a tensor / list / scalar into a python int list (or None)."""
    if value is None:
        return None
    if torch.is_tensor(value):
        return [int(x) for x in value.detach().cpu().reshape(-1).tolist()]
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(value)]


def _num_heads_q(q_shape, layout_q):
    """BNSD: (B,N,S,D) -> dim1 ; BSND: (B,S,N,D) -> dim2 ; TND: dim1."""
    if layout_q == "BSND":
        return int(q_shape[2])
    return int(q_shape[1])


def _num_heads_kv(k_shape, layout_kv):
    """Mirror pytest `_resolve_axis_layout` for the KV layouts."""
    if "PA" in layout_kv:
        if layout_kv == "PA_NZ":
            return int(k_shape[1])
        if layout_kv == "PA_BBND":
            return int(k_shape[2])
        if layout_kv == "PA_BNBD":
            return int(k_shape[1])
        return int(k_shape[1])
    if layout_kv == "TND":
        return int(k_shape[1])
    return int(k_shape[-2]) if len(k_shape) >= 2 else 1


def _head_dim(q_shape, layout_q):
    return int(q_shape[-1])


def _batch_size(q_shape, layout_q, cu_seqlens_q, seqused_q, kwargs):
    """BNSD/BSND use batch from the explicit attribute or ``q.shape[0]``; TND infers
    from cu_seqlens_q / seqused_q (mirror torch_extension._calculate_batch_size)."""
    if layout_q == "TND":
        cu_q = _to_int_list(cu_seqlens_q)
        if cu_q and len(cu_q) > 0:
            return len(cu_q) - 1
        seq_q = _to_int_list(seqused_q)
        if seq_q:
            return len(seq_q)
        return int(q_shape[0] if len(q_shape) else 0)
    explicit = get_attribute(kwargs, "batch_size")
    if explicit is not None:
        return int(explicit)
    return int(q_shape[0]) if len(q_shape) else 1


def build_metadata_arguments(q, k, cu_seqlens_q, seqused_q, seqused_kv, kwargs):
    """Derive ``mixed_quant_flash_attn_metadata`` arguments from the invocation.

    Returns both the scalar args used to size the metadata and the (possibly via
    kwargs) sidecar seqused/cu_seqlens lists.
    """
    layout_q = str(get_attribute(kwargs, "layout_q", "BSND"))
    layout_kv = str(get_attribute(kwargs, "layout_kv", "PA_BBND"))
    q_shape = tuple(int(x) for x in q.shape)
    k_shape = tuple(int(x) for x in k.shape)

    return {
        "num_heads_q": int(
            get_attribute(kwargs, "num_heads_q", _num_heads_q(q_shape, layout_q))
        ),
        "num_heads_kv": int(
            get_attribute(kwargs, "num_heads_kv", _num_heads_kv(k_shape, layout_kv))
        ),
        "head_dim": int(
            get_attribute(kwargs, "head_dim", _head_dim(q_shape, layout_q))
        ),
        "quant_mode": int(
            get_attribute(
                kwargs, "quant_mode", get_attribute(kwargs, "quant_compute_mode", 1)
            )
        ),
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_q": seqused_q,
        "seqused_kv": seqused_kv,
        "batch_size": int(
            _batch_size(q_shape, layout_q, cu_seqlens_q, seqused_q, kwargs)
        ),
        "max_seqlen_q": int(get_attribute(kwargs, "max_seqlen_q", -1)),
        "max_seqlen_kv": int(get_attribute(kwargs, "max_seqlen_kv", -1)),
        "mask_mode": int(get_attribute(kwargs, "mask_mode", 0)),
        "win_left": int(get_attribute(kwargs, "win_left", -1)),
        "win_right": int(get_attribute(kwargs, "win_right", -1)),
        "layout_q": layout_q,
        "layout_kv": layout_kv,
        "layout_attn_out": str(get_attribute(kwargs, "layout_attn_out", layout_q)),
    }


def _move_to_device(value, anchor):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device=anchor.device)
    return torch.as_tensor(_to_int_list(value), dtype=torch.int32, device=anchor.device)


def run_metadata(arguments, anchor):
    """Invoke the companion metadata op on device, anchored on ``anchor``."""
    import cann_ops_transformer  # noqa: F401  (ensure op registration)

    return torch.ops.cann_ops_transformer.mixed_quant_flash_attn_metadata(
        num_heads_q=int(arguments["num_heads_q"]),
        num_heads_kv=int(arguments["num_heads_kv"]),
        head_dim=int(arguments["head_dim"]),
        quant_mode=int(arguments["quant_mode"]),
        cu_seqlens_q=_move_to_device(arguments["cu_seqlens_q"], anchor),
        seqused_q=_move_to_device(arguments["seqused_q"], anchor),
        seqused_kv=_move_to_device(arguments["seqused_kv"], anchor),
        batch_size=int(arguments["batch_size"]),
        max_seqlen_q=int(arguments["max_seqlen_q"]),
        max_seqlen_kv=int(arguments["max_seqlen_kv"]),
        mask_mode=int(arguments["mask_mode"]),
        win_left=int(arguments["win_left"]),
        win_right=int(arguments["win_right"]),
        layout_q=str(arguments["layout_q"]),
        layout_kv=str(arguments["layout_kv"]),
        layout_attn_out=str(arguments["layout_attn_out"]),
    )


def run(
    q,
    k,
    v,
    k_descale,
    v_descale,
    block_table=None,
    cu_seqlens_q=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    attn_mask=None,
    metadata=None,
    quant_mode=1,
    softmax_scale=1.0,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="PA_BBND",
    layout_attn_out="BSND",
    return_softmax_lse=False,
    **kwargs,
):
    """Materialize the metadata slot after H2D, before the main API/timing.

    Signature mirrors ``torch.ops.cann_ops_transformer.mixed_quant_flash_attn``
    so TTK delivers the op-schema-ordered positional args; sidecar attributes
    (batch_size / num_heads_q / num_heads_kv / head_dim / quant_compute_mode ...)
    arrive via kwargs.

    Dynamic slot (CSV ``tensor_view_shapes`` carries ``(-1,)``, metadata is None):
    return ``{"metadata": generated}`` for TTK to back-fill the main call.
    Static slot (existing placeholder tensor): resize_/copy_ in place and return None.
    """
    # 位置标量参数桥接进 kwargs，供 build_metadata_arguments 统一读取。
    hook_kwargs = dict(kwargs)
    hook_kwargs.setdefault("quant_mode", quant_mode)
    hook_kwargs.setdefault("mask_mode", mask_mode)
    hook_kwargs.setdefault("max_seqlen_q", max_seqlen_q)
    hook_kwargs.setdefault("max_seqlen_kv", max_seqlen_kv)
    hook_kwargs.setdefault("layout_q", layout_q)
    hook_kwargs.setdefault("layout_kv", layout_kv)
    hook_kwargs.setdefault("layout_attn_out", layout_attn_out)
    hook_kwargs.setdefault("win_left", win_left)
    hook_kwargs.setdefault("win_right", win_right)

    arguments = build_metadata_arguments(
        q, k, cu_seqlens_q, seqused_q, seqused_kv, hook_kwargs
    )

    if metadata is None:
        generated = run_metadata(arguments, q)
        return {"metadata": generated}

    generated = run_metadata(arguments, metadata)
    if tuple(metadata.shape) != tuple(generated.shape):
        # 静态占位槽位大小可能与生成的实际大小不一致（如固定 16384 占位 vs 实际 4096）。
        # 主算子按 metadata op 算出的实际大小读取（_calculate_metadata_size），因此
        # 统一把占位张量 resize 到生成大小再回填，确保传给主算子的 metadata 尺寸正确。
        metadata.resize_(generated.shape)
    metadata.copy_(generated.to(dtype=metadata.dtype, device=metadata.device))
    return None
