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


"""TTK npu_preprocess plugin for quant_flash_mla.

Generates metadata via the companion metadata op and returns it
so TTK fills the metadata kw_only slot before calling the main op.
"""

import logging

import torch

logger = logging.getLogger(__name__)


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
    layout_q="BSND",
    layout_kv="PA_BNBD",
    layout_out="BSND",
    return_softmax_lse=False,
    **kwargs,
):
    """Generate metadata for the main quant_flash_mla call."""
    testcase_name = kwargs.get("testcase_name", "unknown")
    logging.info(
        "[%s] build QuantMla metadata (quant_mode=%s)", testcase_name, quant_mode
    )

    N_q = kwargs.get("N_q")
    N_kv = kwargs.get("N_kv", 1)
    head_dim_qk = kwargs.get("D", 512) + kwargs.get("D_rope", 64)  # 576

    # cu_seqlens_q: only for TND layout; pass None for non-TND layouts.
    # C++ binding uses c10::optional<at::Tensor>, so None is properly handled.
    cu_seqlens_q_t = None
    if (
        cu_seqlens_q is not None
        and hasattr(cu_seqlens_q, "numel")
        and cu_seqlens_q.numel() > 0
    ):
        cu_seqlens_q_t = cu_seqlens_q.to(torch.int32).npu()
    elif kwargs.get("cu_seqlens_q_values") is not None:
        vals = kwargs["cu_seqlens_q_values"]
        cu_seqlens_q_t = torch.tensor(vals, dtype=torch.int32, device="npu")

    # seqused_q: from kw_only tensor or sidecar values
    seqused_q_t = None
    if seqused_q is not None and hasattr(seqused_q, "numel") and seqused_q.numel() > 0:
        seqused_q_t = seqused_q.to(torch.int32).npu()
    elif kwargs.get("seqused_q_values") is not None:
        vals = kwargs["seqused_q_values"]
        seqused_q_t = torch.tensor(vals, dtype=torch.int32, device="npu")

    # cache_seqlens: move to NPU int32
    cache_seqlens_t = cache_seqlens.to(torch.int32).npu()

    # max_seqlen auto-derive
    if max_seqlen_q is None or max_seqlen_q <= 0:
        if kwargs.get("seqused_q_values") is not None:
            max_seqlen_q = max(kwargs["seqused_q_values"])
        elif seqused_q_t is not None:
            max_seqlen_q = int(seqused_q_t.max().item())
        else:
            max_seqlen_q = int(q.shape[2]) if q.dim() == 4 else int(q.shape[0])

    if max_seqlen_kv is None or max_seqlen_kv <= 0:
        if kwargs.get("cache_seqlens_values") is not None:
            max_seqlen_kv = max(kwargs["cache_seqlens_values"])
        else:
            max_seqlen_kv = int(cache_seqlens_t.max().item())

    md = torch.ops.cann_ops_transformer.quant_flash_mla_metadata(
        cache_seqlens=cache_seqlens_t,
        num_heads_q=N_q,
        num_heads_kv=N_kv,
        quant_mode=quant_mode,
        cu_seqlens_q=cu_seqlens_q_t,
        seqused_q=seqused_q_t,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_kv=max_seqlen_kv,
        head_dim_qk=head_dim_qk,
        head_dim_v=head_dim_v,
        mask_mode=mask_mode,
        layout_q=layout_q,
    )
    logging.info("[%s] metadata generated, shape=%s", testcase_name, tuple(md.shape))

    # 动态槽位 (CSV tensor_view_shapes 含 -1, ttk 传入 None): 按参数名物化并返回,
    # 由 ttk 回填主算子调用;
    # 静态槽位 (已有 Tensor): 原地 resize_/copy_ 回填并返回 None。
    if metadata is None:
        logging.info(
            "[%s] materialize dynamic metadata slot: shape=%s",
            testcase_name,
            tuple(md.shape),
        )
        return {"metadata": md}

    if tuple(metadata.shape) == (0,):
        metadata.resize_(md.shape)
    if metadata.numel() != md.numel():
        raise ValueError(
            "QuantMla metadata size mismatch: "
            f"placeholder={tuple(metadata.shape)} (numel={metadata.numel()}), "
            f"generated={tuple(md.shape)} (numel={md.numel()})"
        )
    if tuple(metadata.shape) != tuple(md.shape):
        md = md.reshape(metadata.shape)
    metadata.copy_(md.to(dtype=metadata.dtype, device=metadata.device))
    return None
