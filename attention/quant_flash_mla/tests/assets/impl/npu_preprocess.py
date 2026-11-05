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
"""Materialize only explicit metadata placeholders without repairing inputs.

TTK calls this hook after H2D. Main-op tensors are authoritative; *_values
are written by inputs.py, never reconstructed here. None / int32 (0,) / a
zero-filled int32 (2, N) buffer request metadata generation in auto mode.
Nonzero metadata and malformed shapes/dtypes are passed through unchanged.
"""

import logging
import torch

logger = logging.getLogger(__name__)


def get_attribute(kwargs, name, default=None):
    value = kwargs.get(name)
    if value is None:
        value = kwargs.get(f"pytest_{name}")
    return default if value is None else value


def _dimension(tensor, axis):
    if (
        not torch.is_tensor(tensor)
        or axis is None
        or not -tensor.ndim <= axis < tensor.ndim
    ):
        return None
    return tensor.shape[axis]


def _omitted_sequence(tensor):
    # Match only the documented legacy int32 (0,) slot convention when
    # deriving a dense scheduling length; preserve every actual main input.
    return tensor is None or (
        torch.is_tensor(tensor)
        and tensor.dtype == torch.int32
        and tuple(tensor.shape) == (0,)
    )


def build_metadata_arguments(q, cu_seqlens_q, seqused_q, kwargs, k_cache=None):
    """Infer missing metadata attributes from real layouts, keeping overrides."""
    layout_q = kwargs.get("layout_q", "TND")
    layout_kv = kwargs.get("layout_kv", "PA_BNBD")
    q_axis = {"TND": 1, "BNSD": 1, "BSND": 2}.get(layout_q)
    kv_axis = {"PA_BNBD": 1, "PA_BBND": 2, "PA_NZ": 1}.get(layout_kv)
    arguments = {
        "num_heads_q": get_attribute(
            kwargs, "num_heads_q", get_attribute(kwargs, "N_q", _dimension(q, q_axis))
        ),
        "num_heads_kv": get_attribute(
            kwargs,
            "num_heads_kv",
            get_attribute(kwargs, "N_kv", _dimension(k_cache, kv_axis)),
        ),
        "head_dim_qk": get_attribute(
            kwargs, "head_dim_qk", get_attribute(kwargs, "D", _dimension(q, -1))
        ),
        "head_dim_v": kwargs.get("head_dim_v", 512),
        "quant_mode": kwargs.get("quant_mode", 1),
        "mask_mode": kwargs.get("mask_mode", 0),
        "layout_q": layout_q,
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_q": seqused_q,
        "max_seqlen_q": kwargs.get("max_seqlen_q", -1),
        "max_seqlen_kv": kwargs.get("max_seqlen_kv", -1),
    }
    # Dense Q with no length tensors means all S rows are used. Metadata has
    # no q input, so give it that equivalent length; leave explicit bounds
    # (including invalid negatives) and the main-op arguments untouched.
    if (
        layout_q in ("BSND", "BNSD")
        and torch.is_tensor(q)
        and q.ndim == 4
        and _omitted_sequence(cu_seqlens_q)
        and _omitted_sequence(seqused_q)
        and arguments["max_seqlen_q"] in (None, -1)
    ):
        s_axis = 1 if layout_q == "BSND" else 2
        arguments["seqused_q"] = torch.full(
            (q.shape[0],), q.shape[s_axis], dtype=torch.int32, device=q.device
        )
    return arguments


def run_metadata(arguments, cache_seqlens):
    import cann_ops_transformer  # noqa: F401 (register actual Torch operator)

    # Already prepared by TTK. Do not cast dtype, move to a default NPU,
    # repair a sequence, or round a scalar before calling the real API.
    return torch.ops.cann_ops_transformer.quant_flash_mla_metadata(
        cache_seqlens=cache_seqlens, **arguments
    )


def _metadata_slot_kind(metadata):
    if metadata is None:
        return "dynamic"
    if not torch.is_tensor(metadata) or metadata.dtype != torch.int32:
        return "explicit"
    if tuple(metadata.shape) == (0,):
        return "legacy"
    if metadata.ndim != 2 or metadata.shape[0] != 2 or metadata.shape[1] <= 0:
        return "explicit"
    return "buffer" if torch.count_nonzero(metadata).item() == 0 else "explicit"


def _copy_metadata_payload(metadata, generated):
    """Materialize an auto placeholder only when its shape matches the API output."""
    if tuple(metadata.shape) != tuple(generated.shape):
        raise ValueError(
            f"[PREPROCESS] metadata placeholder {tuple(metadata.shape)} != generated {tuple(generated.shape)}; "
            "use a dynamic slot, or metadata_policy=preserve for a raw metadata test"
        )
    metadata.copy_(generated)


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
    softmax_scale=None,
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
    """Generate metadata only for recognized slots; never replace main inputs.

    metadata_policy=preserve bypasses generation even for None/zero slots,
    for replay or metadata-negative tests. Default auto recognizes only the
    placeholder forms described in the module docstring.
    """
    policy = kwargs.get("metadata_policy", "auto")
    if policy not in ("auto", "preserve"):
        raise ValueError("metadata_policy must be 'auto' or 'preserve'")
    kind = _metadata_slot_kind(metadata) if policy == "auto" else "explicit"
    if kind == "explicit":
        logger.info(
            "[%s] keeping supplied metadata for main-op validation/execution",
            kwargs.get("testcase_name", "unknown"),
        )
        return None
    # Bound schema arguments are authoritative over sidecar/pytest aliases.
    attributes = dict(kwargs)
    attributes.update(
        quant_mode=quant_mode,
        mask_mode=mask_mode,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        head_dim_v=head_dim_v,
        layout_q=layout_q,
        layout_kv=layout_kv,
    )
    arguments = build_metadata_arguments(
        q, cu_seqlens_q, seqused_q, attributes, k_cache=k_cache
    )
    generated = run_metadata(arguments, cache_seqlens)
    if kind == "dynamic":
        return {"metadata": generated}
    if kind == "legacy":
        metadata.resize_(generated.shape)
    _copy_metadata_payload(metadata, generated)
    return None
