# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import sys
import types
from typing import Optional

import torch

from .indexer_prologue_k import (  # noqa: F401
    indexer_prologue_k as _indexer_prologue_k_impl,
)

# The on-disk directory has a _dsl suffix. Attach the public operator to the
# shared ds41 module and to cann_ops_transformer.ops explicitly.
_ops_mod = None
for _key in ("cann_ops_transformer.ops", "cann_ops_transformer_custom.ops"):
    _ops_mod = sys.modules.get(_key)
    if _ops_mod is not None:
        break
if _ops_mod is None:
    raise ImportError(
        "cann_ops_transformer.ops is not loaded; import it before "
        "indexer_prologue_k_dsl"
    )

parent_pkg = _ops_mod.__name__
module_name = f"{parent_pkg}.ds41"
ds41 = sys.modules.get(module_name)
if ds41 is None:
    ds41 = types.ModuleType(module_name)
    ds41.__package__ = parent_pkg
    sys.modules[module_name] = ds41


def indexer_prologue_k(
    latent: torch.Tensor,
    wk: torch.Tensor,
    norm_weight: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
    k_cache: torch.Tensor,
    k_scale_cache: Optional[torch.Tensor] = None,
    *,
    cache_index: torch.Tensor,
    storage_mode: int,
    norm_eps: float,
    combined_block_size: int = -1,
) -> torch.Tensor:
    """Run the eager alias ABI or its graph-safe mutation-only counterpart."""
    if torch.compiler.is_compiling():
        torch.ops.cann_ops_transformer.ds41._indexer_prologue_k_inplace(
            latent,
            wk,
            norm_weight,
            rope_sin,
            rope_cos,
            k_cache,
            k_scale_cache,
            cache_index=cache_index,
            storage_mode=storage_mode,
            norm_eps=norm_eps,
            combined_block_size=combined_block_size,
        )
        return k_cache
    return torch.ops.cann_ops_transformer.ds41.indexer_prologue_k(
        latent,
        wk,
        norm_weight,
        rope_sin,
        rope_cos,
        k_cache,
        k_scale_cache,
        cache_index=cache_index,
        storage_mode=storage_mode,
        norm_eps=norm_eps,
        combined_block_size=combined_block_size,
    )


# Single-op custom wheels generate ``attention/__init__.py`` with an import
# matching the on-disk directory name. Keep that packaging-only alias valid.
indexer_prologue_k_dsl = indexer_prologue_k
ds41.indexer_prologue_k = indexer_prologue_k
_ops_mod.ds41 = ds41
_ops_mod.indexer_prologue_k = indexer_prologue_k

__all__ = ["ds41", "indexer_prologue_k", "indexer_prologue_k_dsl"]
