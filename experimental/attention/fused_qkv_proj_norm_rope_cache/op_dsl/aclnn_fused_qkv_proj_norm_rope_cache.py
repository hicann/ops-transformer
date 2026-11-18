# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
from opkit import aclnn

from .fused_qkv_proj_norm_rope_cache import (
    fused_qkv_proj_norm_rope_cache as _implementation,
)


@aclnn("FusedQkvProjNormRopeCache")
def fused_qkv_proj_norm_rope_cache(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    k_gamma: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    eps: float,
    positions_contiguous: bool,
):
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
    empty = torch.empty((0, 256), dtype=q.dtype, device=q.device)
    return q, k, v, empty, empty.clone()
