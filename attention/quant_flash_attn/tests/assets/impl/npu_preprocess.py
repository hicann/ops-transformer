#!/usr/bin/python3
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
"""Populate the QuantFlashAttn metadata slot before the main API call.

This module is intentionally independent from TTK.  The framework passes the
main API arguments after H2D; this hook invokes the companion torch operator
``quant_flash_attn_metadata`` and updates ``metadata`` in place.  Profiling and
result handling remain in TTK, and the hook always returns ``None``.

The metadata construction was extracted from the two-phase ``npu_*_fa`` golden
implementations (``npu_mxfp8_fa`` / ``npu_gqa_fp8_fa`` / ``npu_hif8_fa``) that
previously called ``quant_flash_attn_metadata`` followed by ``quant_flash_attn``.
"""

import logging

import torch


def get_attribute(kwargs, name, default=None):
    """Resolve an attribute from ``kwargs`` with an optional ``pytest_`` alias."""
    value = kwargs.get(name)
    if value is None:
        value = kwargs.get(f"pytest_{name}")
    return default if value is None else value


def get_values(kwargs, name):
    """Read a ``*_values`` sidecar list, if present, into a python list."""
    value = kwargs.get(f"{name}_values")
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().reshape(-1).tolist()
    return [int(item) for item in value]


def _resolve_tensor(tensor, kwargs, name):
    """把 cu_seqlens/seqused 位置参数归一成 None 或有效 tensor。

    直调 op 后, 这些位置参数可能是空 (0,) 张量 (CSV 空 slot 生成); 当对应的
    ``*_values`` 属性也为 None (未填写) 时, 归一成 None 而非空张量——否则
    torch_extension 的 metadata 实现 (quant_flash_attn.py 的 _calculate_batch_size)
    会把空张量的 size(0)=0 误当 batch_size。参考 flash_attn 资产的 _resolve_tensor。
    """
    if tensor is not None and int(torch.as_tensor(tensor).numel()) > 0:
        return tensor
    values = get_values(kwargs, name)
    if values is None:
        return None
    return torch.as_tensor(values, dtype=torch.int32)


def _num_heads_q(q_shape, layout_q):
    if layout_q == "BSND":
        return int(q_shape[2])
    return int(q_shape[1])


def _num_heads_kv(k_shape, layout_kv):
    if layout_kv == "PA_BBND":
        return int(k_shape[2])
    return int(k_shape[1])


def _derive_batch_size(q_shape, layout_q, cu_seqlens_q, seqused_q, kwargs):
    """Priority mirrors torch_extension._calculate_batch_size.

    seqused_q.numel() -> cu_seqlens_q.numel() - 1 -> explicit batch_size ->
    shape fallback.  The ``*_values`` lists sit between the live tensors and the
    explicit attribute for compatibility with TTK small-integer delivery.
    """
    if seqused_q is not None:
        return int(torch.as_tensor(seqused_q).numel())
    if cu_seqlens_q is not None:
        return max(int(torch.as_tensor(cu_seqlens_q).numel()) - 1, 0)
    seq_q_values = get_values(kwargs, "seqused_q")
    if seq_q_values is not None:
        return len(seq_q_values)
    cu_q_values = get_values(kwargs, "cu_seqlens_q")
    if cu_q_values is not None:
        return max(len(cu_q_values) - 1, 0)
    explicit = get_attribute(kwargs, "batch_size")
    if explicit is not None:
        return int(explicit)
    if layout_q == "BSND":
        return int(q_shape[0])
    return 0


def _max_seqlen(kwargs, name):
    """透传 CSV 的 max_seqlen_q/kv, 空则 -1。

    metadata 的 max_seqlen_q/kv 必须与主算子 quant_flash_attn 拿到的值一致:
    主算子由 ttk 按 CSV 属性直调 (空 → None → -1), 因此这里不能从 seqused/cu_seqlens
    派生真实值, 否则 metadata 与主算子对序列边界的理解不一致, 导致 npu 输出错误。
    """
    explicit = get_attribute(kwargs, f"max_seqlen_{name}")
    if explicit is not None:
        return int(explicit)
    return -1


def build_metadata_arguments(
    q, k, v, cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, kwargs
):
    """Derive ``quant_flash_attn_metadata`` arguments from the main invocation."""
    layout_q = str(get_attribute(kwargs, "layout_q", "BSND"))
    layout_q_descale = str(get_attribute(kwargs, "layout_q_descale", layout_q))
    layout_kv = str(get_attribute(kwargs, "layout_kv", "BSND"))
    layout_out = str(get_attribute(kwargs, "layout_out", layout_q))

    q_shape = tuple(int(value) for value in q.shape)
    k_shape = tuple(int(value) for value in k.shape)
    head_dim = int(
        get_attribute(
            kwargs,
            "head_dim",
            get_attribute(kwargs, "D", q_shape[-1]),
        )
    )

    num_heads_q = int(
        get_attribute(
            kwargs,
            "num_heads_q",
            get_attribute(kwargs, "N_q", _num_heads_q(q_shape, layout_q)),
        )
    )
    num_heads_kv = int(
        get_attribute(
            kwargs,
            "num_heads_kv",
            get_attribute(kwargs, "N_kv", _num_heads_kv(k_shape, layout_kv)),
        )
    )

    is_tnd_q = layout_q == "TND"
    is_tnd_kv = layout_kv == "TND"

    # Mirror the original two-phase golden implementations:
    #   - cu_seqlens only feeds the metadata op for TND layouts (otherwise None),
    #     and batch_size is only provided for non-TND layouts.
    # 空 (0,) 张量归一成 None, 否则 torch_extension 重算 batch_size 时会把
    # size(0)=0 误当 batch (参考 flash_attn 资产的 _resolve_tensor)。
    cu_q = _resolve_tensor(cu_seqlens_q, kwargs, "cu_seqlens_q") if is_tnd_q else None
    cu_kv = (
        _resolve_tensor(cu_seqlens_kv, kwargs, "cu_seqlens_kv") if is_tnd_kv else None
    )
    seq_q = _resolve_tensor(seqused_q, kwargs, "seqused_q")
    seq_kv = _resolve_tensor(seqused_kv, kwargs, "seqused_kv")
    batch_size = (
        None if is_tnd_q else _derive_batch_size(q_shape, layout_q, cu_q, seq_q, kwargs)
    )

    return {
        "num_heads_q": num_heads_q,
        "num_heads_kv": num_heads_kv,
        "head_dim": head_dim,
        "quant_mode": int(get_attribute(kwargs, "quant_mode", 1)),
        "cu_seqlens_q": cu_q,
        "cu_seqlens_kv": cu_kv,
        "seqused_q": seq_q,
        "seqused_kv": seq_kv,
        "batch_size": None if batch_size is None else int(batch_size),
        "max_seqlen_q": _max_seqlen(kwargs, "q"),
        "max_seqlen_kv": _max_seqlen(kwargs, "kv"),
        "head_dim_v": get_attribute(kwargs, "head_dim_v"),
        "mask_mode": int(get_attribute(kwargs, "mask_mode", 0)),
        "layout_q": layout_q,
        "layout_q_descale": layout_q_descale,
        "layout_kv": layout_kv,
        "layout_out": layout_out,
    }


def move_to_device(value, target):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device=target.device)
    return torch.as_tensor(value, device=target.device)


def run_metadata(arguments, metadata):
    import cann_ops_transformer  # noqa: F401  (ensure op registration)

    return torch.ops.cann_ops_transformer.quant_flash_attn_metadata(
        int(arguments["num_heads_q"]),
        int(arguments["num_heads_kv"]),
        int(arguments["head_dim"]),
        int(arguments["quant_mode"]),
        cu_seqlens_q=move_to_device(arguments.get("cu_seqlens_q"), metadata),
        cu_seqlens_kv=move_to_device(arguments.get("cu_seqlens_kv"), metadata),
        seqused_q=move_to_device(arguments.get("seqused_q"), metadata),
        seqused_kv=move_to_device(arguments.get("seqused_kv"), metadata),
        batch_size=None
        if arguments.get("batch_size") is None
        else int(arguments["batch_size"]),
        max_seqlen_q=int(arguments["max_seqlen_q"]),
        max_seqlen_kv=int(arguments["max_seqlen_kv"]),
        head_dim_v=(
            None
            if arguments.get("head_dim_v") is None
            else int(arguments["head_dim_v"])
        ),
        mask_mode=int(arguments["mask_mode"]),
        layout_q=str(arguments["layout_q"]),
        layout_q_descale=str(arguments["layout_q_descale"]),
        layout_kv=str(arguments["layout_kv"]),
        layout_out=str(arguments["layout_out"]),
    )


def run(
    q,
    k,
    v,
    q_descale,
    k_descale,
    v_descale,
    quant_mode,
    block_table=None,
    p_scale=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    attn_mask=None,
    metadata=None,
    softmax_scale=1.0,
    mask_mode=0,
    win_left=-1,
    win_right=-1,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_q_descale="BSND",
    layout_kv="BSND",
    layout_out="BSND",
    return_softmax_lse=False,
    **kwargs,
):
    """Generate and copy metadata for a main QuantFlashAttn invocation.

    Signature mirrors ``torch.ops.cann_ops_transformer.quant_flash_attn`` so TTK
    delivers the op-schema-ordered positional args; extra sidecar attributes
    (batch_size / N_q / N_kv / D / head_dim_v / *_values ...) arrive via kwargs.
    """
    if metadata is None:
        raise ValueError("QuantFlashAttn npu_preprocess requires metadata")
    if quant_mode is None:
        quant_mode = get_attribute(kwargs, "quant_mode", 1)
    # 位置标量参数桥接进 kwargs, 供 build_metadata_arguments 统一读取。
    hook_kwargs = dict(kwargs)
    hook_kwargs.setdefault("quant_mode", quant_mode)
    hook_kwargs.setdefault("mask_mode", mask_mode)
    hook_kwargs.setdefault("max_seqlen_q", max_seqlen_q)
    hook_kwargs.setdefault("max_seqlen_kv", max_seqlen_kv)
    hook_kwargs.setdefault("layout_q", layout_q)
    hook_kwargs.setdefault("layout_q_descale", layout_q_descale)
    hook_kwargs.setdefault("layout_kv", layout_kv)
    hook_kwargs.setdefault("layout_out", layout_out)
    testcase_name = hook_kwargs.get("testcase_name")
    logging.info(
        "[%s] build QuantFlashAttn metadata (quant_mode=%s)", testcase_name, quant_mode
    )

    arguments = build_metadata_arguments(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_kv,
        seqused_q,
        seqused_kv,
        hook_kwargs,
    )
    generated = run_metadata(arguments, metadata)
    if tuple(metadata.shape) != tuple(generated.shape):
        raise ValueError(
            "QuantFlashAttn metadata shape mismatch: "
            f"placeholder={tuple(metadata.shape)}, generated={tuple(generated.shape)}"
        )
    metadata.copy_(generated.to(dtype=metadata.dtype, device=metadata.device))
    return None
