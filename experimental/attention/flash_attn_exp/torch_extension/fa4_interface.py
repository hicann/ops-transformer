# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""FA4 flash_attn_func compatibility adapter for flash_attn_exp.

The current backend implements only dense, unmasked forward attention. Unsupported
FA4 features are rejected before invoking the backend.
"""

import torch

from .flash_attn_exp import flash_attn_exp, flash_attn_exp_metadata


def flash_attn_func(
    q,
    k,
    v,
    qv=None,
    gather_kv_indices=None,
    softmax_scale=None,
    causal=False,
    window_size=(None, None),
    learnable_sink=None,
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    deterministic=False,
    score_mod=None,
    score_mod_bwd=None,
    mask_mod=None,
    aux_tensors=None,
    aux_scalars=None,
    block_sparse_tensors=None,
    block_sparse_tensors_bwd=None,
    return_lse=False,
    gather_bwd_recompute_p=False,
    gather_bwd_token_chunk=None,
    config=None,
    scales=None,
):
    """FA4-compatible dense attention entry point; returns (output, LSE or None).

    Only NPU FP16/BF16 inference, with BSND inputs and head dimension 64,
    is supported. The underlying operator and metadata share the same shapes.
    """
    unsupported = {
        "qv": qv is not None,
        "gather_kv_indices": gather_kv_indices is not None,
        "causal": causal,
        "window_size": window_size != (None, None),
        "learnable_sink": learnable_sink is not None,
        "softcap": softcap != 0.0,
        "num_splits": num_splits != 1,
        "pack_gqa": pack_gqa is not None,
        "deterministic": deterministic,
        "score_mod": score_mod is not None,
        "score_mod_bwd": score_mod_bwd is not None,
        "mask_mod": mask_mod is not None,
        "aux_tensors": aux_tensors is not None,
        "aux_scalars": aux_scalars is not None,
        "block_sparse_tensors": block_sparse_tensors is not None,
        "block_sparse_tensors_bwd": block_sparse_tensors_bwd is not None,
        "gather_bwd_recompute_p": gather_bwd_recompute_p,
        "gather_bwd_token_chunk": gather_bwd_token_chunk is not None,
        "config": config is not None,
        "scales": scales is not None,
    }
    rejected = [name for name, enabled in unsupported.items() if enabled]
    if rejected:
        raise NotImplementedError(
            "flash_attn_exp does not support: " + ", ".join(rejected)
        )
    if torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v)):
        raise NotImplementedError("flash_attn_exp does not support backward")
    if any(t.ndim != 4 for t in (q, k, v)):
        raise ValueError("q, k and v must have BSND layout (four dimensions)")
    if any(t.device.type != "npu" for t in (q, k, v)) or not (
        q.device == k.device == v.device
    ):
        raise ValueError("q, k and v must be on the same NPU device")
    if (
        q.dtype not in (torch.float16, torch.bfloat16)
        or q.dtype != k.dtype
        or q.dtype != v.dtype
    ):
        raise ValueError("q, k and v must have the same float16 or bfloat16 dtype")
    if q.shape[-1] != 64 or k.shape[-1] != 64 or v.shape[-1] != 64:
        raise ValueError(
            "flash_attn_exp supports head_dim and head_dim_v equal to 64 only"
        )
    if k.shape != v.shape or q.shape[0] != k.shape[0]:
        raise ValueError(
            "k and v must have identical shapes and the same batch size as q"
        )
    if any(dim <= 0 for t in (q, k, v) for dim in t.shape) or q.shape[0] >= 65536:
        raise ValueError(
            "all dimensions must be positive and batch size must be less than 65536"
        )
    if q.shape[2] % k.shape[2] != 0:
        raise ValueError(
            "the number of query heads must be divisible by the number of KV heads"
        )
    if not k.is_contiguous() or not v.is_contiguous():
        raise ValueError("k and v must be contiguous")
    batch, seqlen_q, heads_q, head_dim = q.shape
    seqlen_k, heads_k = k.shape[1:3]
    metadata = flash_attn_exp_metadata(
        heads_q,
        heads_k,
        head_dim,
        batch_size=batch,
        max_seqlen_q=seqlen_q,
        max_seqlen_kv=seqlen_k,
        mask_mode=0,
        layout_q="BSND",
        layout_kv="BSND",
        layout_out="BSND",
    )
    out, lse = flash_attn_exp(
        q,
        k,
        v,
        metadata=metadata,
        softmax_scale=head_dim**-0.5 if softmax_scale is None else softmax_scale,
        mask_mode=0,
        max_seqlen_q=seqlen_q,
        max_seqlen_kv=seqlen_k,
        layout_q="BSND",
        layout_kv="BSND",
        layout_out="BSND",
        return_softmax_lse=return_lse,
    )
    return out, lse if return_lse else None
