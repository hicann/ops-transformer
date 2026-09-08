# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional
import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class KeyPoolOpBuilder(OpBuilder):
    def __init__(self):
        super(KeyPoolOpBuilder, self).__init__("key_pool", category="attention")

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/key_pool.cpp"]

    def schema(self):
        """PyTorch operator signature."""
        return (
            "key_pool(Tensor hidden_states, Tensor wk, Tensor gate_weight, Tensor ape, Tensor(a!) state_cache, "
            "Tensor cache_block_table, Tensor start_pos, *, "
            "Tensor? norm_weight=None, Tensor? norm_bias=None, Tensor? cos=None, Tensor? sin=None, "
            "Tensor? cu_seqlens=None, Tensor? seqused=None, int cmp_ratio=4, float norm_eps=1e-6, "
            "int rotary_mode=1) -> Tensor"
        )

    def register_meta(self):
        """
        Registers the Meta implementation (Shape/Dtype inference).
        Essential for Autograd and FakeTensor support.
        """

        @impl(get_as_library(), self.name, "Meta")
        def key_pool_meta(
            hidden_states,
            wk,
            gate_weight,
            ape,
            state_cache,
            cache_block_table,
            start_pos,
            *,
            norm_weight=None,
            norm_bias=None,
            cos=None,
            sin=None,
            cu_seqlens=None,
            seqused=None,
            cmp_ratio=4,
            norm_eps=1e-6,
            rotary_mode=1,
        ):
            if (norm_weight is None) != (norm_bias is None):
                raise ValueError("norm_weight and norm_bias must be passed as a pair")
            if (cos is None) != (sin is None):
                raise ValueError("cos and sin must be passed as a pair")
            if cos is not None:
                raise NotImplementedError(
                    "KeyPool RoPE is not implemented in this stage"
                )
            if hidden_states.dim() == 2 and cu_seqlens is None:
                raise ValueError("cu_seqlens is required for TH layout")
            if hidden_states.dim() == 3 and cu_seqlens is not None:
                raise ValueError("cu_seqlens must be absent for BSH layout")
            if seqused is not None:
                raise ValueError("seqused is reserved and must be None in this stage")
            b = cache_block_table.size(0)
            pcap = (
                cache_block_table.size(1) * state_cache.size(1) + cmp_ratio - 1
            ) // cmp_ratio
            pooled_key_size = (b, pcap, wk.size(0))

            return torch.empty(
                pooled_key_size, dtype=hidden_states.dtype, device="meta"
            )


key_pool_op_builder = KeyPoolOpBuilder()
key_pool_op_builder._ensure_initialized()


@impl(get_as_library(), key_pool_op_builder.name, "PrivateUse1")
def key_pool(
    hidden_states: torch.Tensor,
    wk: torch.Tensor,
    gate_weight: torch.Tensor,
    ape: torch.Tensor,
    state_cache: torch.Tensor,
    cache_block_table: torch.Tensor,
    start_pos: torch.Tensor,
    *,
    norm_weight: Optional[torch.Tensor] = None,
    norm_bias: Optional[torch.Tensor] = None,
    cos: Optional[torch.Tensor] = None,
    sin: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    cmp_ratio: int = 4,
    norm_eps: float = 1e-6,
    rotary_mode: int = 1,
) -> torch.Tensor:
    """Run the KeyPool operator on NPU tensors.

    Args:
        hidden_states: Input hidden states in BSH ``[B, S, H]`` or TH
            ``[T, H]`` layout. It is an ND, contiguous Tensor with float16 or
            bfloat16 dtype.
        wk: K projection weight with shape ``[D, H]``. It is an ND,
            contiguous Tensor with the same dtype as ``hidden_states``.
        gate_weight: Gate projection weight with shape ``[D, H]``. It is an
            ND, contiguous Tensor with the same dtype as ``hidden_states``.
        ape: Position bias with shape ``[cmp_ratio, D]``. It is an ND,
            contiguous Tensor with float32 dtype.
        state_cache: In-place cache with shape
            ``[block_num, block_size, 2 * D]`` and float32 dtype. It is an ND
            Tensor; the first axis may be non-contiguous and its actual
            stride is passed to the backend. The first ``D`` channels store K
            and the remaining ``D`` channels store Gate.
        cache_block_table: Logical-to-physical cache block table with shape
            ``[B, L]``, int32 dtype, and contiguous ND layout.
        start_pos: Per-batch logical start positions with shape ``[B]`` and
            int32 dtype. It is a contiguous ND Tensor.
        norm_weight: Optional float32 LayerNorm weight with shape ``[D]``.
            It is a contiguous ND Tensor and must be provided together with
            ``norm_bias``.
        norm_bias: Optional float32 LayerNorm bias with shape ``[D]``. Must
            be provided together with ``norm_weight``.
        cos: Optional RoPE cosine input. Must be omitted in this version.
        sin: Optional RoPE sine input. Must be omitted in this version.
        cu_seqlens: Optional contiguous ND int32 prefix sums with shape
            ``[B + 1]``. It is required for TH layout and must be omitted for
            BSH layout.
        seqused: Reserved optional input. Must be omitted in this version.
        cmp_ratio: Compression ratio. Supports ``2, 4, 8, 16, 32, 64, 128``;
            default is ``4``.
        norm_eps: LayerNorm epsilon, must be greater than zero; default is
            ``1e-6``.
        rotary_mode: RoPE mode, supports ``0`` or ``1``; default is ``1``.
            RoPE computation is not implemented in this version.

    Returns:
        The pooled key tensor with shape ``[B, Sr, D]`` and the same dtype as
        ``hidden_states``. ``state_cache`` is updated in place.
    """
    if (norm_weight is None) != (norm_bias is None):
        raise ValueError("norm_weight and norm_bias must be passed as a pair")
    if (cos is None) != (sin is None):
        raise ValueError("cos and sin must be passed as a pair")
    if cos is not None:
        raise NotImplementedError("KeyPool RoPE is not implemented in this stage")
    if hidden_states.dim() == 2 and cu_seqlens is None:
        raise ValueError("cu_seqlens is required for TH layout")
    if hidden_states.dim() == 3 and cu_seqlens is not None:
        raise ValueError("cu_seqlens must be absent for BSH layout")
    if seqused is not None:
        raise ValueError("seqused is reserved and must be None in this stage")
    op_module = key_pool_op_builder.load()
    return op_module.key_pool(
        hidden_states,
        wk,
        gate_weight,
        ape,
        state_cache,
        cache_block_table,
        start_pos,
        norm_weight,
        norm_bias,
        cos,
        sin,
        cu_seqlens,
        seqused,
        cmp_ratio,
        norm_eps,
        rotary_mode,
    )
