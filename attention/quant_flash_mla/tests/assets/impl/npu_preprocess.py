#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Populate the QuantFlashMla metadata slot before the main API call.

This module is intentionally independent from TTK.  The framework passes the
main API arguments after H2D; this hook invokes the companion torch operator
``quant_flash_mla_metadata`` and updates ``metadata`` in place.  Profiling and
result handling remain in TTK, and the hook always returns ``None`` for static
slots or ``{"metadata": generated}`` for dynamic slots (CSV -1 placeholder).

The metadata construction was extracted from the two-phase golden
implementations that previously called ``quant_flash_mla_metadata`` followed by
``quant_flash_mla``; structure mirrors
``quant_flash_attn/tests/assets/impl/npu_preprocess.py``.
"""

import logging

import torch


logger = logging.getLogger(__name__)


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
    metadata 算子会把空张量的 size(0)=0 误当 batch_size。
    参考 quant_flash_attn/tests/assets/impl/npu_preprocess.py 的 _resolve_tensor。
    """
    if tensor is not None and int(torch.as_tensor(tensor).numel()) > 0:
        return tensor
    values = get_values(kwargs, name)
    if values is None:
        return None
    return torch.as_tensor(values, dtype=torch.int32)


def _max_seqlen(kwargs, name):
    """透传 CSV 的 max_seqlen_q/kv, 空则 -1。

    metadata 的 max_seqlen_q/kv 必须与主算子 quant_flash_mla 拿到的值一致:
    主算子由 ttk 按 CSV 属性直调 (空 → None → -1), 因此这里不能从 seqused/cu_seqlens
    派生真实值, 否则 metadata 与主算子对序列边界的理解不一致, 导致 npu 输出错误。
    参考 quant_flash_attn/tests/assets/impl/npu_preprocess.py 的 _max_seqlen。
    """
    explicit = get_attribute(kwargs, f"max_seqlen_{name}")
    if explicit is not None:
        return int(explicit)
    return -1


def build_metadata_arguments(q, cu_seqlens_q, seqused_q, kwargs):
    """Derive ``quant_flash_mla_metadata`` arguments from the main invocation."""
    # 不同 CSV 对 'D' 属性的语义不一致:
    #   qfmla_abnormal_full.csv: D = nope 部分 (512), 需要加 D_rope (64) 才得到 576
    #   B270_redline.csv:        D = QK 总 head dim (576), 不应再加 D_rope
    # 仿照 quant_flash_attn/tests/assets/impl/npu_preprocess.py 的 head_dim 解析:
    #   ① 优先 head_dim_qk 显式属性 (异常用例可原样透传被算子拦截)
    #   ② 退化到 Q 张量最后一维 (永远等于真实 head_dim_qk)
    head_dim_qk = int(
        get_attribute(
            kwargs,
            "head_dim_qk",
            q.shape[-1],
        )
    )

    num_heads_q = int(
        get_attribute(
            kwargs,
            "N_q",
            get_attribute(kwargs, "num_heads_q", q.shape[1]),
        )
    )
    num_heads_kv = int(get_attribute(kwargs, "N_kv", 1))

    # 空 (0,) 张量归一成 None, 否则 metadata 算子重算 batch_size 时会把
    # size(0)=0 误当 batch (参考 quant_flash_attn 资产的 _resolve_tensor)。
    cu_q = _resolve_tensor(cu_seqlens_q, kwargs, "cu_seqlens_q")
    seq_q = _resolve_tensor(seqused_q, kwargs, "seqused_q")

    return {
        "num_heads_q": num_heads_q,
        "num_heads_kv": num_heads_kv,
        "head_dim_qk": head_dim_qk,
        "quant_mode": int(get_attribute(kwargs, "quant_mode", 1)),
        "cu_seqlens_q": cu_q,
        "seqused_q": seq_q,
        "max_seqlen_q": _max_seqlen(kwargs, "q"),
        "max_seqlen_kv": _max_seqlen(kwargs, "kv"),
        "head_dim_v": get_attribute(kwargs, "head_dim_v"),
        "mask_mode": int(get_attribute(kwargs, "mask_mode", 0)),
        "layout_q": str(get_attribute(kwargs, "layout_q", "TND")),
    }


def move_to_device(value, target):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.to(device=target.device)
    return torch.as_tensor(value, device=target.device)


def run_metadata(arguments, cache_seqlens):
    """Call quant_flash_mla_metadata using device-anchored tensors.

    ``cache_seqlens`` is the main op's cache_seqlens input (already on device),
    used both as the metadata op's required first positional input and as the
    device anchor for tensors materialized from CSV ``*_values`` sidecars.
    """
    import cann_ops_transformer  # noqa: F401  (ensure op registration)

    return torch.ops.cann_ops_transformer.quant_flash_mla_metadata(
        cache_seqlens=cache_seqlens,
        num_heads_q=int(arguments["num_heads_q"]),
        num_heads_kv=int(arguments["num_heads_kv"]),
        quant_mode=int(arguments["quant_mode"]),
        cu_seqlens_q=move_to_device(arguments.get("cu_seqlens_q"), cache_seqlens),
        seqused_q=move_to_device(arguments.get("seqused_q"), cache_seqlens),
        max_seqlen_q=int(arguments["max_seqlen_q"]),
        max_seqlen_kv=int(arguments["max_seqlen_kv"]),
        head_dim_qk=int(arguments["head_dim_qk"]),
        head_dim_v=(
            None
            if arguments.get("head_dim_v") is None
            else int(arguments["head_dim_v"])
        ),
        mask_mode=int(arguments["mask_mode"]),
        layout_q=str(arguments["layout_q"]),
    )


def run(
    q,
    k_cache,
    q_descale,
    k_descale,
    block_table,
    cache_seqlens,
    quant_mode,
    *,
    cu_seqlens_q=None,
    seqused_q=None,
    attn_mask=None,
    metadata=None,
    softmax_scale=1.0,
    mask_mode=0,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    head_dim_v=512,
    layout_q="TND",
    layout_kv="PA_BNBD",
    layout_out="TND",
    return_softmax_lse=False,
    **kwargs,
):
    """Materialize metadata after H2D, before timing and graph capture.

    Signature mirrors ``torch.ops.cann_ops_transformer.quant_flash_mla`` so TTK
    delivers the op-schema-ordered positional args; extra sidecar attributes
    (N_q / N_kv / D / head_dim_v / *_values ...) arrive via kwargs.

    动态槽位 (CSV tensor_view_shapes 含 -1, ttk 传 metadata=None): 按参数名返回
    ``{"metadata": generated}``, 由 ttk 回填主算子 args/kwargs;
    静态槽位 (旧 (0,) 占位或已有 Tensor): 原地 resize_/copy_ 回填并隐式返回 None。
    """
    if quant_mode is None:
        quant_mode = get_attribute(kwargs, "quant_mode", 1)
    # 位置标量参数桥接进 kwargs, 供 build_metadata_arguments 统一读取。
    hook_kwargs = dict(kwargs)
    hook_kwargs.setdefault("quant_mode", quant_mode)
    hook_kwargs.setdefault("mask_mode", mask_mode)
    hook_kwargs.setdefault("max_seqlen_q", max_seqlen_q)
    hook_kwargs.setdefault("max_seqlen_kv", max_seqlen_kv)
    hook_kwargs.setdefault("head_dim_v", head_dim_v)
    hook_kwargs.setdefault("layout_q", layout_q)
    testcase_name = hook_kwargs.get("testcase_name", "unknown")
    logging.info(
        "[%s] build QuantMla metadata (quant_mode=%s)", testcase_name, quant_mode
    )

    arguments = build_metadata_arguments(q, cu_seqlens_q, seqused_q, hook_kwargs)
    cache_seqlens_t = cache_seqlens.npu()
    # run_metadata 需要 device 锚点 tensor, 用 cache_seqlens (主算子必传输入)。
    generated = run_metadata(arguments, cache_seqlens_t)
    logging.info(
        "[%s] metadata generated, shape=%s", testcase_name, tuple(generated.shape)
    )

    # 动态槽位 (CSV tensor_view_shapes 含 -1, ttk 传入 None): 按参数名物化并返回,
    # 由 ttk 回填主算子调用;
    # 静态槽位 (已有 Tensor): 原地 resize_/copy_ 回填并返回 None。
    if metadata is None:
        logging.info(
            "[%s] materialize dynamic metadata slot: shape=%s",
            testcase_name,
            tuple(generated.shape),
        )
        return {"metadata": generated}

    # 与 quant_flash_attn/tests/assets/impl/npu_preprocess.py 对齐:
    # 旧 (0,) 占位 -> resize 到 generated.shape;
    # 其它 shape 不一致直接报错, 不 silently reshape——CSV 配错 metadata shape
    # 必须被显式暴露, 不应被 plugin 掩盖。
    if tuple(metadata.shape) == (0,):
        metadata.resize_(generated.shape)
    if tuple(metadata.shape) != tuple(generated.shape):
        raise ValueError(
            "QuantMla metadata shape mismatch: "
            f"placeholder={tuple(metadata.shape)}, generated={tuple(generated.shape)}"
        )
    metadata.copy_(generated.to(dtype=metadata.dtype, device=metadata.device))
    return None
