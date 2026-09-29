# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""PyTorch dispatcher adapter for the ds41 CANNBotDSL indexer K prologue."""

from typing import Optional

import torch
import torch_npu  # noqa: F401 - initializes the PrivateUse1 backend
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library


_INPLACE_OP_NAME = "ds41._indexer_prologue_k_inplace"


class IndexerPrologueKOpBuilder(OpBuilder):
    def __init__(self):
        super(IndexerPrologueKOpBuilder, self).__init__(
            "ds41.indexer_prologue_k", category="attention"
        )

    def sources(self):
        """The implementation is supplied by the CANNBotDSL net-ops wheel."""
        return []

    def schema(self):
        """Register eager and functionalization-safe graph schemas."""
        return [
            (
                "ds41.indexer_prologue_k("
                "Tensor latent, Tensor wk, Tensor norm_weight, Tensor rope_sin, "
                "Tensor rope_cos, Tensor(a!) k_cache, Tensor(b!)? k_scale_cache=None, "
                "*, Tensor cache_index, int storage_mode, float norm_eps, "
                "int combined_block_size=-1) -> Tensor(a!)"
            ),
            (
                "ds41._indexer_prologue_k_inplace("
                "Tensor latent, Tensor wk, Tensor norm_weight, Tensor rope_sin, "
                "Tensor rope_cos, Tensor(a!) k_cache, Tensor(b!)? k_scale_cache=None, "
                "*, Tensor cache_index, int storage_mode, float norm_eps, "
                "int combined_block_size=-1) -> ()"
            ),
        ]

    def register_meta(self):
        """The operator returns the same cache object that it mutates."""

        @impl(get_as_library(), self.name, "Meta")
        def indexer_prologue_k_meta(
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
            return k_cache

        @impl(get_as_library(), _INPLACE_OP_NAME, "Meta")
        def indexer_prologue_k_inplace_meta(
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
        ) -> None:
            return None


indexer_prologue_k_op_builder = IndexerPrologueKOpBuilder()
indexer_prologue_k_op_builder._ensure_initialized()


def _run_indexer_prologue_k(
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
    """Run the packaged CANNBotDSL kernel shared by both dispatcher schemas."""
    from ops.indexer_prologue_k import indexer_prologue_k as dsl_indexer_prologue_k

    return dsl_indexer_prologue_k(
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


@impl(get_as_library(), indexer_prologue_k_op_builder.name, "PrivateUse1")
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
    """Eager dispatcher ABI that returns the mutated ``k_cache`` alias."""
    return _run_indexer_prologue_k(
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


@impl(get_as_library(), _INPLACE_OP_NAME, "PrivateUse1")
def indexer_prologue_k_inplace(
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
) -> None:
    """Graph ABI without an aliasing output, for AOT functionalization."""
    _run_indexer_prologue_k(
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
    return None
