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
"""Populate QuantSparseFlashMla metadata without a TTK context."""

import importlib.util
import logging
import sys
from pathlib import Path

import torch


OPERATOR = "quant_sparse_flash_mla"
METADATA_INDEX = 20
ACLNN_PARAMETER_NAMES = (
    "q",
    "ori_kv",
    "cmp_kv",
    "q_descale",
    "ori_kv_descale",
    "cmp_kv_descale",
    "ori_sparse_indices",
    "cmp_sparse_indices",
    "ori_block_table",
    "cmp_block_table",
    "cu_seqlens_q",
    "cu_seqlens_ori_kv",
    "cu_seqlens_cmp_kv",
    "seqused_q",
    "seqused_ori_kv",
    "seqused_cmp_kv",
    "cmp_residual_kv",
    "ori_topk_length",
    "cmp_topk_length",
    "sinks",
    "metadata",
    "quant_mode",
    "softmax_scale",
    "cmp_ratio",
    "ori_mask_mode",
    "cmp_mask_mode",
    "ori_win_left",
    "ori_win_right",
    "layout_q",
    "layout_kv",
    "topk_value_mode",
    "return_softmax_lse",
    "attn_out",
    "softmax_lse_out",
)


def load_metadata_protocol():
    name = "qsmla_ttk_metadata_protocol"
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


def max_sequence(prefix, used, fallback, *, cumulative=False):
    if used is not None:
        values = torch.as_tensor(used).detach().reshape(-1)
        if values.numel():
            return int(values.max().item())
    if prefix is not None:
        values = torch.as_tensor(prefix).detach().reshape(-1)
        if values.numel() > 1:
            if cumulative:
                return int(values.max().item())
            return int((values[1:] - values[:-1]).max().item())
    return int(fallback)


def build_metadata_arguments(q, ori_kv, cmp_kv, quant_mode, kwargs):
    layout_q = str(get_attribute(kwargs, "layout_q", "BSND"))
    layout_kv = str(get_attribute(kwargs, "layout_kv", "BSND"))
    q_shape = tuple(int(value) for value in q.shape)
    cu_q = kwargs.get("cu_seqlens_q")
    cu_ori = kwargs.get("cu_seqlens_ori_kv")
    cu_cmp = kwargs.get("cu_seqlens_cmp_kv")
    seq_q = kwargs.get("seqused_q")
    seq_ori = kwargs.get("seqused_ori_kv")
    seq_cmp = kwargs.get("seqused_cmp_kv")
    cu_q_values = get_values(kwargs, "cu_seqlens_q")
    cu_ori_values = get_values(kwargs, "cu_seqlens_ori_kv")
    cu_cmp_values = get_values(kwargs, "cu_seqlens_cmp_kv")
    seq_q_values = get_values(kwargs, "seqused_q")
    seq_ori_values = get_values(kwargs, "seqused_ori_kv")
    seq_cmp_values = get_values(kwargs, "seqused_cmp_kv")

    num_heads_q = q_shape[2] if layout_q == "BSND" else q_shape[1]
    if ori_kv is None:
        num_heads_kv, ori_fallback = 0, 0
    elif layout_kv == "TND":
        num_heads_kv, ori_fallback = int(ori_kv.shape[1]), int(ori_kv.shape[0])
    elif layout_kv == "PA_BBND":
        num_heads_kv = int(ori_kv.shape[2])
        ori_fallback = int(get_attribute(kwargs, "S2", ori_kv.shape[1]))
    else:
        num_heads_kv, ori_fallback = int(ori_kv.shape[2]), int(ori_kv.shape[1])

    if cmp_kv is None:
        cmp_fallback = 0
    elif layout_kv == "TND":
        cmp_fallback = int(cmp_kv.shape[0])
    elif layout_kv == "PA_BBND":
        cmp_fallback = int(get_attribute(kwargs, "S2", cmp_kv.shape[1]))
        cmp_fallback //= max(int(get_attribute(kwargs, "cmp_ratio", 1) or 1), 1)
    else:
        cmp_fallback = int(cmp_kv.shape[1])

    batch_size = get_attribute(kwargs, "batch_size")
    if batch_size is None:
        if seq_q_values is not None:
            batch_size = len(seq_q_values)
        elif cu_q_values is not None:
            batch_size = len(cu_q_values) - 1
        elif seq_q is not None:
            batch_size = int(torch.as_tensor(seq_q).numel())
        elif cu_q is not None:
            batch_size = int(torch.as_tensor(cu_q).numel()) - 1
        else:
            batch_size = q_shape[0] if layout_q == "BSND" else 0

    ori_sparse = kwargs.get("ori_sparse_indices")
    cmp_sparse = kwargs.get("cmp_sparse_indices")
    ori_topk = 0 if ori_sparse is None else int(ori_sparse.shape[-1])
    cmp_topk = 0 if cmp_sparse is None else int(cmp_sparse.shape[-1])
    q_fallback = q_shape[1] if layout_q == "BSND" else q_shape[0]
    max_q = get_attribute(kwargs, "max_seqlen_q")
    if max_q is None:
        max_q = max_sequence(
            cu_q_values if cu_q_values is not None else cu_q,
            seq_q_values if seq_q_values is not None else seq_q,
            q_fallback,
            cumulative=layout_q == "TND",
        )
    max_ori = max_sequence(
        cu_ori_values if cu_ori_values is not None else cu_ori,
        seq_ori_values if seq_ori_values is not None else seq_ori,
        ori_fallback,
    )
    max_cmp = max_sequence(
        cu_cmp_values if cu_cmp_values is not None else cu_cmp,
        seq_cmp_values if seq_cmp_values is not None else seq_cmp,
        cmp_fallback,
    )
    return {
        "num_heads_q": int(get_attribute(kwargs, "num_heads_q", num_heads_q)),
        "num_heads_kv": int(get_attribute(kwargs, "num_heads_kv", num_heads_kv)),
        "head_dim": int(get_attribute(kwargs, "head_dim", q_shape[-1])),
        "quant_mode": int(quant_mode),
        "cu_seqlens_q": cu_q,
        "cu_seqlens_ori_kv": cu_ori,
        "cu_seqlens_cmp_kv": cu_cmp,
        "seqused_q": seq_q,
        "seqused_ori_kv": seq_ori,
        "seqused_cmp_kv": seq_cmp,
        "cmp_residual_kv": kwargs.get("cmp_residual_kv"),
        "ori_topk_length": kwargs.get("ori_topk_length"),
        "cmp_topk_length": kwargs.get("cmp_topk_length"),
        "batch_size": int(batch_size),
        "max_seqlen_q": int(max_q),
        "max_seqlen_ori_kv": int(get_attribute(kwargs, "max_seqlen_ori_kv", max_ori)),
        "max_seqlen_cmp_kv": int(get_attribute(kwargs, "max_seqlen_cmp_kv", max_cmp)),
        "ori_topk": int(
            get_attribute(
                kwargs, "metadata_ori_topk", get_attribute(kwargs, "ori_topk", ori_topk)
            )
        ),
        "topk": int(
            get_attribute(
                kwargs, "metadata_cmp_topk", get_attribute(kwargs, "cmp_topk", cmp_topk)
            )
        ),
        "cmp_ratio": int(get_attribute(kwargs, "cmp_ratio", 1) or 1),
        "ori_mask_mode": int(get_attribute(kwargs, "ori_mask_mode", 4)),
        "cmp_mask_mode": int(get_attribute(kwargs, "cmp_mask_mode", 3)),
        "ori_win_left": int(get_attribute(kwargs, "ori_win_left", 127)),
        "ori_win_right": int(get_attribute(kwargs, "ori_win_right", 0)),
        "layout_q": layout_q,
        "layout_kv": layout_kv,
        "has_ori_kv": bool(get_attribute(kwargs, "has_ori_kv", ori_kv is not None)),
        "has_cmp_kv": bool(get_attribute(kwargs, "has_cmp_kv", cmp_kv is not None)),
    }


def move_to_device(value, target, empty_if_none=False):
    if value is None:
        if empty_if_none:
            return torch.empty(0, dtype=torch.int32, device=target.device)
        return None
    if torch.is_tensor(value):
        return value.to(device=target.device)
    return torch.as_tensor(value, device=target.device)


def run_metadata(arguments, metadata):
    import cann_ops_transformer

    return torch.ops.cann_ops_transformer.quant_sparse_flash_mla_metadata(
        int(arguments["num_heads_q"]),
        int(arguments["num_heads_kv"]),
        int(arguments["head_dim"]),
        int(arguments["quant_mode"]),
        cu_seqlens_q=move_to_device(arguments.get("cu_seqlens_q"), metadata, True),
        cu_seqlens_ori_kv=move_to_device(
            arguments.get("cu_seqlens_ori_kv"), metadata, True
        ),
        cu_seqlens_cmp_kv=move_to_device(
            arguments.get("cu_seqlens_cmp_kv"), metadata, True
        ),
        seqused_q=move_to_device(arguments.get("seqused_q"), metadata, True),
        seqused_ori_kv=move_to_device(arguments.get("seqused_ori_kv"), metadata),
        seqused_cmp_kv=move_to_device(arguments.get("seqused_cmp_kv"), metadata),
        cmp_residual_kv=move_to_device(arguments.get("cmp_residual_kv"), metadata),
        ori_topk_length=move_to_device(arguments.get("ori_topk_length"), metadata),
        cmp_topk_length=move_to_device(arguments.get("cmp_topk_length"), metadata),
        batch_size=int(arguments["batch_size"]),
        max_seqlen_q=int(arguments["max_seqlen_q"]),
        max_seqlen_ori_kv=int(arguments["max_seqlen_ori_kv"]),
        max_seqlen_cmp_kv=int(arguments["max_seqlen_cmp_kv"]),
        ori_topk=int(arguments["ori_topk"]),
        cmp_topk=int(arguments["topk"]),
        cmp_ratio=int(arguments["cmp_ratio"]),
        ori_mask_mode=int(arguments["ori_mask_mode"]),
        cmp_mask_mode=int(arguments["cmp_mask_mode"]),
        ori_win_left=int(arguments["ori_win_left"]),
        ori_win_right=int(arguments["ori_win_right"]),
        layout_q=str(arguments["layout_q"]),
        layout_kv=str(arguments["layout_kv"]),
        has_ori_kv=bool(arguments["has_ori_kv"]),
        has_cmp_kv=bool(arguments["has_cmp_kv"]),
    )


def run(q, *, ori_kv=None, cmp_kv=None, metadata=None, quant_mode=None, **kwargs):
    """Generate metadata once, or reuse a nonzero manual-data slot."""
    if metadata is None:
        raise ValueError("QuantSparseFlashMla npu_preprocess requires metadata")
    if quant_mode is None:
        raise ValueError("QuantSparseFlashMla npu_preprocess requires quant_mode")
    protocol = load_metadata_protocol()
    testcase_name = kwargs.get("testcase_name")
    force_metadata_refresh = bool(get_attribute(kwargs, "metadata_refresh", False))
    if protocol.metadata_is_materialized(metadata) and not force_metadata_refresh:
        logging.info("[%s] reuse nonzero QSMLA metadata input", testcase_name)
        return None
    arguments = protocol.load_metadata_inputs(OPERATOR, testcase_name)
    if arguments is not None:
        source = "manual-data sidecar"
    else:
        arguments = build_metadata_arguments(q, ori_kv, cmp_kv, quant_mode, kwargs)
        source = "main API fallback (sidecar unavailable)"
    logging.info(
        "[%s] build QSMLA metadata from %s; forced=%s",
        testcase_name,
        source,
        force_metadata_refresh,
    )
    generated = run_metadata(arguments, metadata)
    if tuple(metadata.shape) != tuple(generated.shape):
        raise ValueError(
            "QSMLA metadata shape mismatch: "
            f"placeholder={tuple(metadata.shape)}, generated={tuple(generated.shape)}"
        )
    metadata.copy_(generated.to(dtype=metadata.dtype, device=metadata.device))
    rewritten = protocol.rewrite_metadata_input(
        OPERATOR, testcase_name, METADATA_INDEX, metadata
    )
    if rewritten is not None:
        logging.info("[%s] rewrote QSMLA metadata input: %s", testcase_name, rewritten)
    return None


def run_aclnn(*args, **kwargs):
    """Adapt the ACLNN main API order to the shared Torch metadata hook."""
    if len(args) != len(ACLNN_PARAMETER_NAMES):
        raise ValueError(
            f"QuantSparseFlashMla ACLNN hook expects {len(ACLNN_PARAMETER_NAMES)} arguments, got {len(args)}"
        )
    values = dict(zip(ACLNN_PARAMETER_NAMES, args))
    host_metadata = values["metadata"]
    metadata = move_to_device(host_metadata, torch.empty(0, device="npu"))
    values["metadata"] = metadata
    q = values.pop("q")
    result = run(q, **values, **kwargs)
    if torch.is_tensor(host_metadata):
        if host_metadata.device != metadata.device:
            host_metadata.copy_(
                metadata.to(dtype=host_metadata.dtype, device=host_metadata.device)
            )
    else:
        host_metadata[...] = metadata.detach().cpu().numpy()
    return result
