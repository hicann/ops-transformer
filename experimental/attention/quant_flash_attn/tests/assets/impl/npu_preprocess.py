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
"""Populate the QuantFlashAttn (MXFP4) metadata slot before the main API call.

本模块独立于 TTK 框架。框架在主 API 调用前 (H2D 之后) 把主算子入参传给本 hook;
hook 调用伴随算子 ``quant_flash_attn_metadata`` 并把结果 ``copy_`` 回填到 metadata
槽位, 随后主算子 ``torch.ops.cann_ops_transformer.quant_flash_attn`` 由 ttk 直调。
hook 始终返回 ``None``。
"""

import logging

import torch


def get_attribute(kwargs, name, default=None):
    """Resolve an attribute from ``kwargs`` with an optional ``pytest_`` alias."""
    value = kwargs.get(name)
    if value is None:
        value = kwargs.get(f"pytest_{name}")
    return default if value is None else value


def _num_heads_q(q_shape, layout_q):
    if layout_q == "BSND":
        return int(q_shape[2])
    return int(q_shape[1])


def _num_heads_kv(k_shape, layout_kv):
    if layout_kv == "PA_BBND":
        return int(k_shape[2])
    return int(k_shape[1])


def _move_to_device(value, target):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device=target.device)
    return torch.as_tensor(value, device=target.device)


def build_metadata_arguments(q, k, v, quant_mode, kwargs):
    """从主算子入参派生 ``quant_flash_attn_metadata`` 的入参 (与 golden 侧一致)。"""
    layout_q = str(get_attribute(kwargs, "layout_q", "BSND"))
    layout_q_descale = str(get_attribute(kwargs, "layout_q_descale", layout_q))
    layout_kv = str(get_attribute(kwargs, "layout_kv", "BSND"))
    layout_out = str(get_attribute(kwargs, "layout_out", layout_q))

    q_shape = tuple(int(value) for value in q.shape)
    k_shape = tuple(int(value) for value in k.shape)

    head_dim = int(
        get_attribute(kwargs, "head_dim", get_attribute(kwargs, "D", q_shape[-1]))
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

    return {
        "num_heads_q": num_heads_q,
        "num_heads_kv": num_heads_kv,
        "head_dim": head_dim,
        "quant_mode": int(quant_mode)
        if quant_mode is not None
        else int(get_attribute(kwargs, "quant_mode", 5)),
        "batch_size": get_attribute(kwargs, "batch_size", get_attribute(kwargs, "B")),
        "max_seqlen_q": int(get_attribute(kwargs, "max_seqlen_q", -1)),
        "max_seqlen_kv": int(get_attribute(kwargs, "max_seqlen_kv", -1)),
        "head_dim_v": get_attribute(
            kwargs, "head_dim_v", get_attribute(kwargs, "V_D", head_dim)
        ),
        "mask_mode": int(get_attribute(kwargs, "mask_mode", 0)),
        "win_left": int(get_attribute(kwargs, "win_left", -1)),
        "win_right": int(get_attribute(kwargs, "win_right", -1)),
        "layout_q": layout_q,
        "layout_q_descale": layout_q_descale,
        "layout_kv": layout_kv,
        "layout_out": layout_out,
        "is_grad_enabled": False,
    }


def _resolve_metadata_op():
    try:
        from cann_ops_transformer.ops import quant_flash_attn_metadata

        return quant_flash_attn_metadata
    except ImportError:
        import torch_npu  # noqa: F401  (ensure op registration)

        return torch.ops.cann_ops_transformer.quant_flash_attn_metadata


def run_metadata(
    arguments, cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, metadata
):
    return _resolve_metadata_op()(
        int(arguments["num_heads_q"]),
        int(arguments["num_heads_kv"]),
        int(arguments["head_dim"]),
        int(arguments["quant_mode"]),
        cu_seqlens_q=_move_to_device(cu_seqlens_q, metadata),
        cu_seqlens_kv=_move_to_device(cu_seqlens_kv, metadata),
        seqused_q=_move_to_device(seqused_q, metadata),
        seqused_kv=_move_to_device(seqused_kv, metadata),
        batch_size=(
            None
            if arguments.get("batch_size") is None
            else int(arguments["batch_size"])
        ),
        max_seqlen_q=int(arguments["max_seqlen_q"]),
        max_seqlen_kv=int(arguments["max_seqlen_kv"]),
        head_dim_v=(
            None
            if arguments.get("head_dim_v") is None
            else int(arguments["head_dim_v"])
        ),
        mask_mode=int(arguments["mask_mode"]),
        win_left=int(arguments["win_left"]),
        win_right=int(arguments["win_right"]),
        layout_q=str(arguments["layout_q"]),
        layout_q_descale=str(arguments["layout_q_descale"]),
        layout_kv=str(arguments["layout_kv"]),
        layout_out=str(arguments["layout_out"]),
        is_grad_enabled=bool(arguments["is_grad_enabled"]),
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
    if metadata is None:
        raise ValueError("QuantFlashAttn (MXFP4) npu_preprocess requires metadata")

    hook_kwargs = dict(kwargs)
    hook_kwargs.setdefault("quant_mode", quant_mode)
    hook_kwargs.setdefault("mask_mode", mask_mode)
    hook_kwargs.setdefault("win_left", win_left)
    hook_kwargs.setdefault("win_right", win_right)
    hook_kwargs.setdefault("max_seqlen_q", max_seqlen_q)
    hook_kwargs.setdefault("max_seqlen_kv", max_seqlen_kv)
    hook_kwargs.setdefault("layout_q", layout_q)
    hook_kwargs.setdefault("layout_q_descale", layout_q_descale)
    hook_kwargs.setdefault("layout_kv", layout_kv)
    hook_kwargs.setdefault("layout_out", layout_out)

    testcase_name = hook_kwargs.get("testcase_name")
    logging.info("[%s] build QuantFlashAttn (MXFP4) metadata", testcase_name)

    arguments = build_metadata_arguments(q, k, v, quant_mode, hook_kwargs)
    generated = run_metadata(
        arguments, cu_seqlens_q, cu_seqlens_kv, seqused_q, seqused_kv, metadata
    )
    if tuple(metadata.shape) != tuple(generated.shape):
        raise ValueError(
            "QuantFlashAttn (MXFP4) metadata shape mismatch: "
            f"placeholder={tuple(metadata.shape)}, generated={tuple(generated.shape)}"
        )
    metadata.copy_(generated.to(dtype=metadata.dtype, device=metadata.device))
    return None
