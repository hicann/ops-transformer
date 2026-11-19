# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import get_as_library

get_as_library().define(
    "fused_qkv_proj_norm_rope_cache("
    "Tensor hidden, Tensor weight, Tensor k_gamma, Tensor positions, "
    "Tensor cos_sin_cache, Tensor slot_mapping, Tensor(a!) k_cache, "
    "Tensor(b!) v_cache, *, float eps=1e-06, bool positions_contiguous=True"
    ") -> (Tensor, Tensor, Tensor, Tensor, Tensor)"
)


@impl(get_as_library(), "fused_qkv_proj_norm_rope_cache", "Meta")
def _fused_qkv_proj_norm_rope_cache_meta(
    hidden,
    weight,
    k_gamma,
    positions,
    cos_sin_cache,
    slot_mapping,
    k_cache,
    v_cache,
    *,
    eps=1e-6,
    positions_contiguous=True,
):
    del k_gamma, positions, cos_sin_cache, slot_mapping, k_cache, v_cache
    del eps, positions_contiguous
    tokens = hidden.shape[0]
    n_units = int(weight.shape[0]) // 256
    q = torch.empty((tokens, 6 * 256), dtype=torch.bfloat16, device="meta")
    k = torch.empty((tokens, 256), dtype=torch.bfloat16, device="meta")
    v = torch.empty((tokens, 256), dtype=torch.bfloat16, device="meta")
    if n_units == 10:
        mirror_k = torch.empty((tokens, 256), dtype=torch.bfloat16, device="meta")
        mirror_v = torch.empty((tokens, 256), dtype=torch.bfloat16, device="meta")
    else:
        mirror_k = torch.empty((0, 256), dtype=torch.bfloat16, device="meta")
        mirror_v = torch.empty((0, 256), dtype=torch.bfloat16, device="meta")
    return q, k, v, mirror_k, mirror_v


@impl(get_as_library(), "fused_qkv_proj_norm_rope_cache", "PrivateUse1")
def fused_qkv_proj_norm_rope_cache(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    k_gamma: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    *,
    eps: float = 1e-6,
    positions_contiguous: bool = True,
):
    from .op_dsl.fused_qkv_proj_norm_rope_cache import (
        fused_qkv_proj_norm_rope_cache as _implementation,
    )

    outputs = _implementation(
        hidden,
        weight,
        k_gamma,
        positions,
        cos_sin_cache,
        slot_mapping,
        k_cache,
        v_cache,
        eps=eps,
        return_v=True,
        positions_contiguous=positions_contiguous,
    )
    if len(outputs) == 5:
        return outputs
    q, k, v = outputs
    # Keep the plain-layer mirror outputs rank-2 so Meta and eager agree.
    empty = torch.empty((0, 256), dtype=q.dtype, device=q.device)
    return q, k, v, empty, empty.clone()
