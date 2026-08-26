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
"""Populate the FlashAttn metadata slot before the main API call.

This module is intentionally independent from TTK.  The framework passes the
main API arguments after H2D; this hook invokes the companion torch operator
``flash_attn_metadata`` and updates ``metadata`` in place.  Profiling and
result handling remain in TTK, and the hook always returns ``None``.
"""

import importlib.util
import logging
import sys
from pathlib import Path

import torch


OPERATOR = "flash_attn"
METADATA_INDEX = 10


def load_metadata_protocol():
    """Lazily load the self-contained TTK metadata sidecar protocol."""
    name = "fa_ttk_metadata_protocol"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).with_name("metadata_protocol.py")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot create import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(name, None)
        raise
    return module


def get_attribute(kwargs, name, default=None):
    value = kwargs.get(name)
    if value is None:
        value = kwargs.get(f"pytest_{name}")
    return default if value is None else value


def get_values(kwargs, name):
    value = kwargs.get(f"{name}_values")
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().reshape(-1).tolist()
    return [int(item) for item in value]


def _resolve_tensor(kwargs, name):
    """Prefer the live tensor argument, then build one from ``*_values``."""
    value = kwargs.get(name)
    if value is not None:
        return value
    values = get_values(kwargs, name)
    if values is None:
        return None
    return torch.tensor(values, dtype=torch.int32)


def _derive_batch_size(q_shape, layout_q, cu_seqlens_q, seqused_q, kwargs):
    """Priority mirrors torch_extension._calculate_batch_size.

    seqused_q.size(0) -> cu_seqlens_q.size(0) - 1 -> explicit batch_size;
    the ``*_values`` lists sit between the tensors and the explicit attribute
    because TTK delivers small integer tensors through those attributes.
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


def _kv_length_fallback(layout_kv, k_shape):
    """Shape-based kv length fallback.

    Deliberately never reads the PageAttention block-size dim: for PA layouts
    there is no sound length in the block buffer shape, so a neutral -1 is
    returned and the metadata op resolves the length internally.  In practice
    gen_cases always supplies seqused_kv (or max_seqlen_kv) for PA cases.
    """
    if layout_kv == "TND":
        return int(k_shape[0])
    if layout_kv == "BNSD":
        return int(k_shape[2])
    if layout_kv == "BSND":
        return int(k_shape[1])
    return -1


def _max_seqlen(kwargs, name, used_tensor, used_values, cu_tensor, cu_values, fallback):
    """Resolve one max_seqlen: explicit attr -> max(seqused) -> cu diff -> shape."""
    explicit = get_attribute(kwargs, f"max_seqlen_{name}")
    if explicit is not None:
        return int(explicit)
    used = used_tensor if used_tensor is not None else used_values
    if used is not None:
        return int(torch.as_tensor(used).detach().max().item())
    prefix = cu_tensor if cu_tensor is not None else cu_values
    if prefix is not None:
        values = torch.as_tensor(prefix).detach().reshape(-1)
        if values.numel() > 1:
            return int((values[1:] - values[:-1]).max().item())
    return int(fallback)


def build_metadata_arguments(q, k, v, kwargs):
    """Derive flash_attn_metadata scalar arguments from the main invocation."""
    layout_q = get_attribute(kwargs, "layout_q", "BSND")
    layout_kv = get_attribute(kwargs, "layout_kv", "BSND")
    layout_out = get_attribute(kwargs, "layout_out", layout_q)
    cu_q = kwargs.get("cu_seqlens_q")
    cu_kv = kwargs.get("cu_seqlens_kv")
    seq_q = kwargs.get("seqused_q")
    seq_kv = kwargs.get("seqused_kv")

    q_shape = tuple(int(value) for value in q.shape)
    head_dim = int(q_shape[-1])
    # head derivation mirrors flash_attn_ttk_ops.build_flash_attn_metadata.
    if layout_q == "TND":
        num_heads_q = int(q_shape[1])
    elif layout_q == "BNSD":
        num_heads_q = int(q_shape[1])
    else:
        num_heads_q = int(q_shape[2])

    if layout_kv == "TND":
        num_heads_kv = int(k.shape[1])
    elif layout_kv == "BNSD":
        num_heads_kv = int(k.shape[1])
    elif layout_kv in ("PA_BNBD", "PA_NZ"):
        num_heads_kv = int(k.shape[1])
    elif layout_kv == "PA_BBND":
        num_heads_kv = int(k.shape[2])
    else:
        num_heads_kv = int(k.shape[2])

    k_shape = tuple(int(value) for value in k.shape)
    q_fallback = q_shape[1] if layout_q == "BSND" else q_shape[0]
    kv_fallback = _kv_length_fallback(layout_kv, k_shape)
    cu_q_values = get_values(kwargs, "cu_seqlens_q")
    cu_kv_values = get_values(kwargs, "cu_seqlens_kv")
    seq_q_values = get_values(kwargs, "seqused_q")
    seq_kv_values = get_values(kwargs, "seqused_kv")

    return {
        "num_heads_q": int(get_attribute(kwargs, "num_heads_q", num_heads_q)),
        "num_heads_kv": int(get_attribute(kwargs, "num_heads_kv", num_heads_kv)),
        "head_dim": int(get_attribute(kwargs, "head_dim", head_dim)),
        "cu_seqlens_q": _resolve_tensor(kwargs, "cu_seqlens_q"),
        "cu_seqlens_kv": _resolve_tensor(kwargs, "cu_seqlens_kv"),
        "seqused_q": _resolve_tensor(kwargs, "seqused_q"),
        "seqused_kv": _resolve_tensor(kwargs, "seqused_kv"),
        "batch_size": _derive_batch_size(q_shape, layout_q, cu_q, seq_q, kwargs),
        "max_seqlen_q": _max_seqlen(
            kwargs, "q", seq_q, seq_q_values, cu_q, cu_q_values, q_fallback
        ),
        "max_seqlen_kv": _max_seqlen(
            kwargs, "kv", seq_kv, seq_kv_values, cu_kv, cu_kv_values, kv_fallback
        ),
        "mask_mode": int(get_attribute(kwargs, "mask_mode", 0)),
        "win_left": int(get_attribute(kwargs, "win_left", -1)),
        "win_right": int(get_attribute(kwargs, "win_right", -1)),
        "layout_q": layout_q,
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
    return torch.ops.cann_ops_transformer.flash_attn_metadata(
        int(arguments["num_heads_q"]),
        int(arguments["num_heads_kv"]),
        int(arguments["head_dim"]),
        cu_seqlens_q=move_to_device(arguments.get("cu_seqlens_q"), metadata),
        cu_seqlens_kv=move_to_device(arguments.get("cu_seqlens_kv"), metadata),
        seqused_q=move_to_device(arguments.get("seqused_q"), metadata),
        seqused_kv=move_to_device(arguments.get("seqused_kv"), metadata),
        batch_size=int(arguments["batch_size"]),
        max_seqlen_q=int(arguments["max_seqlen_q"]),
        max_seqlen_kv=int(arguments["max_seqlen_kv"]),
        mask_mode=int(arguments["mask_mode"]),
        win_left=int(arguments["win_left"]),
        win_right=int(arguments["win_right"]),
        layout_q=str(arguments["layout_q"]),
        layout_kv=str(arguments["layout_kv"]),
        layout_out=str(arguments["layout_out"]),
    )


def run(
    q,
    k,
    v,
    *,
    block_table=None,
    cu_seqlens_q=None,
    cu_seqlens_kv=None,
    seqused_q=None,
    seqused_kv=None,
    sinks=None,
    attn_mask=None,
    metadata=None,
    **kwargs,
):
    """Generate and copy metadata for a main FlashAttn invocation."""
    if metadata is None:
        raise ValueError("FlashAttn npu_preprocess requires metadata")
    protocol = load_metadata_protocol()
    testcase_name = kwargs.get("testcase_name")
    if protocol.metadata_is_materialized(metadata):
        # In the E2E flow customize_inputs zeroes the placeholder, so this
        # branch is unreachable there; replay correctness comes from the
        # idempotent re-derivation below, so it is not a bug.
        logging.info("[%s] reuse nonzero FlashAttn metadata input", testcase_name)
        return None

    arguments = protocol.load_metadata_inputs(OPERATOR, testcase_name)
    if arguments is not None:
        source = "manual-data sidecar"
    else:
        arguments = build_metadata_arguments(q, k, v, kwargs)
        source = "main API fallback (sidecar unavailable)"
        saved = protocol.save_metadata_inputs(OPERATOR, testcase_name, arguments)
        if saved is not None:
            logging.info(
                "[%s] saved FlashAttn metadata sidecar: %s", testcase_name, saved
            )
    logging.info("[%s] build FlashAttn metadata from %s", testcase_name, source)
    generated = run_metadata(arguments, metadata)
    if tuple(metadata.shape) != tuple(generated.shape):
        raise ValueError(
            "FlashAttn metadata shape mismatch: "
            f"placeholder={tuple(metadata.shape)}, generated={tuple(generated.shape)}"
        )
    metadata.copy_(generated.to(dtype=metadata.dtype, device=metadata.device))
    rewritten = protocol.rewrite_metadata_input(
        OPERATOR,
        testcase_name,
        METADATA_INDEX,
        metadata,
    )
    if rewritten is not None:
        logging.info(
            "[%s] rewrote FlashAttn metadata input: %s", testcase_name, rewritten
        )
    return None
