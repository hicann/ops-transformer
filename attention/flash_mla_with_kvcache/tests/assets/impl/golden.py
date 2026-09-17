# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""TTK CPU adapter for the supplied ops-transformer-testkit golden."""

import torch
from .flash_mla_with_kvcache_golden import FlashMlaWithKvcacheGolden


def cpu_flash_mla_with_kvcache(
    q,
    k_cache,
    *,
    block_table=None,
    cache_seqlens=None,
    cu_seqlens_q=None,
    seqused_q=None,
    attn_mask=None,
    metadata=None,
    head_dim_v=512,
    softmax_scale=1.0,
    mask_mode=0,
    max_seqlen_q=-1,
    max_seqlen_kv=-1,
    layout_q="BSND",
    layout_kv="PA_BBND",
    layout_out="BSND",
    return_softmax_lse=False,
    **unused,
):
    arguments = dict(locals())
    arguments.pop("unused")
    arguments.pop("metadata")
    for name, value in arguments.items():
        if torch.is_tensor(value):
            arguments[name] = value.detach().cpu()
    out, lse = FlashMlaWithKvcacheGolden().forward_golden(
        **arguments, device_mode="cpu"
    )
    # The kernel does not define the LSE output when it is disabled.
    return out, lse if return_softmax_lse else None
