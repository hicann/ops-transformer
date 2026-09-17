# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Metadata-first TTK entry point; tensor order matches the extension schema."""

from typing import Optional
import torch
import cann_ops_transformer  # Register the extension operators.

import importlib.util
from pathlib import Path


def _build_metadata(**arguments):
    path = Path(__file__).with_name("impl") / "metadata.py"
    spec = importlib.util.spec_from_file_location("_mla_ttk_metadata", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.build_metadata(**arguments)


def flash_mla_with_kvcache_ttk(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    *,
    block_table: Optional[torch.Tensor] = None,
    cache_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    attn_mask: Optional[torch.Tensor] = None,
    metadata: Optional[torch.Tensor] = None,
    head_dim_v: int = 512,
    softmax_scale: float = 1.0,
    mask_mode: int = 0,
    max_seqlen_q: int = -1,
    max_seqlen_kv: int = -1,
    layout_q: str = "BSND",
    layout_kv: str = "PA_BBND",
    layout_out: str = "BSND",
    return_softmax_lse: bool = False,
):
    arguments = dict(locals())
    if metadata is None:
        arguments["metadata"] = _build_metadata(**arguments)
    return torch.ops.cann_ops_transformer.flash_mla_with_kvcache(**arguments)
