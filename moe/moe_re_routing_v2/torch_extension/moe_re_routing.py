# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional, Tuple
import torch
import torch_npu
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


_DTYPE_HIFLOAT8 = torch_npu.hifloat8
_DTYPE_FLOAT4_E2M1 = torch_npu.float4_e2m1fn_x2
_DTYPE_FLOAT4_E1M2 = torch_npu.float4_e1m2fn_x2

TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP = {
    _DTYPE_HIFLOAT8: torch.uint8,
    _DTYPE_FLOAT4_E2M1: torch.uint8,
    _DTYPE_FLOAT4_E1M2: torch.uint8,
}

_TOKENS_DTYPE_ALLOWLIST = (_DTYPE_HIFLOAT8, _DTYPE_FLOAT4_E2M1, _DTYPE_FLOAT4_E1M2)


class _MoeReRoutingOpBuilder(OpBuilder):
    def __init__(self):
        super(_MoeReRoutingOpBuilder, self).__init__("moe_re_routing", category="moe")

    def sources(self):
        return ["csrc/moe/moe_re_routing.cpp"]

    def schema(self) -> str:
        return (
            "moe_re_routing(Tensor tokens, Tensor expert_token_num_per_rank, "
            "*, Tensor? per_token_scales=None, Tensor? expert_topk_weight=None, "
            "int expert_token_num_type=1, int idx_type=0, "
            "int? tokens_dtype=None) "
            "-> (Tensor, Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def moe_re_routing_meta(
            tokens,
            expert_token_num_per_rank,
            per_token_scales=None,
            expert_topk_weight=None,
            expert_token_num_type=1,
            idx_type=0,
            tokens_dtype=None,
        ):
            torch._check(
                tokens.dim() > 1,
                lambda: f"tokens dim should larger than 1, but got {tokens.dim()}D.{ops_error(ErrCode.VALUE)}",
            )
            torch._check(
                expert_token_num_per_rank.dim() > 1,
                lambda: f"expert_token_num_per_rank dim should larger than 1, "
                f"but got {expert_token_num_per_rank.dim()}D.{ops_error(ErrCode.VALUE)}",
            )
            torch._check(
                expert_token_num_type in (0, 1),
                lambda: f"expert_token_num_type must be 0 or 1, got {expert_token_num_type}.{ops_error(ErrCode.VALUE)}",
            )
            torch._check(
                idx_type in (0, 1),
                lambda: f"idx_type must be 0 or 1, got {idx_type}.{ops_error(ErrCode.VALUE)}",
            )
            if tokens_dtype is not None:
                torch._check(
                    tokens_dtype in _TOKENS_DTYPE_ALLOWLIST,
                    lambda: f"tokens_dtype only supports hifloat8, float4_e2m1fn_x2, "
                    f"float4_e1m2fn_x2 or None.{ops_error(ErrCode.VALUE)}",
                )

            permute_tokens_size = list(tokens.size())
            if per_token_scales is None:
                permute_per_token_scales_size = [tokens.size(0)]
                permute_per_token_scales_dtype = torch.float32
            else:
                permute_per_token_scales_size = list(per_token_scales.size())
                permute_per_token_scales_dtype = per_token_scales.dtype

            permute_tokens_dtype = (
                TORCH_DTYPE_ENUM_VALUE_TO_SCALAR_TYPE_MAP.get(
                    tokens_dtype, tokens.dtype
                )
                if tokens_dtype is not None
                else tokens.dtype
            )

            permute_token_idx_size = [tokens.size(0)]
            expert_token_num_size = [expert_token_num_per_rank.size(1)]

            if expert_topk_weight is not None:
                permute_topk_weight = expert_topk_weight.new_empty(
                    expert_topk_weight.size(), dtype=expert_topk_weight.dtype
                )
            else:
                permute_topk_weight = tokens.new_empty((0,), dtype=torch.float32)

            return (
                torch.empty(
                    permute_tokens_size,
                    dtype=permute_tokens_dtype,
                    device=tokens.device,
                ),
                torch.empty(
                    permute_per_token_scales_size,
                    dtype=permute_per_token_scales_dtype,
                    device=tokens.device,
                ),
                torch.empty(
                    permute_token_idx_size, dtype=torch.int32, device=tokens.device
                ),
                torch.empty(
                    expert_token_num_size,
                    dtype=expert_token_num_per_rank.dtype,
                    device=tokens.device,
                ),
                permute_topk_weight,
            )


_moe_re_routing_op_builder = _MoeReRoutingOpBuilder()
_moe_re_routing_op_builder._ensure_initialized()


@impl(get_as_library(), _moe_re_routing_op_builder.name, "PrivateUse1")
def _moe_re_routing(
    tokens,
    expert_token_num_per_rank,
    per_token_scales=None,
    expert_topk_weight=None,
    expert_token_num_type=1,
    idx_type=0,
    tokens_dtype=None,
):
    op_module = _moe_re_routing_op_builder.load()
    return op_module.moe_re_routing(
        tokens,
        expert_token_num_per_rank,
        per_token_scales,
        expert_topk_weight,
        expert_token_num_type,
        idx_type,
        tokens_dtype,
    )


def moe_re_routing(
    tokens: torch.Tensor,
    expert_token_num_per_rank: torch.Tensor,
    per_token_scales: Optional[torch.Tensor] = None,
    expert_topk_weight: Optional[torch.Tensor] = None,
    expert_token_num_type: int = 1,
    idx_type: int = 0,
    tokens_dtype: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """moe_re_routing operator.

    Rearranges tokens from rank-order to expert-order for MoE (Mixture of Experts)
    distributed inference.

    Args:
        tokens: Input tensor of shape (A, H) or multi-dim with first dim A,
            dtype supports float16/bfloat16/int8/float8_e5m2/float8_e4m3fn/
            hifloat8/float4_e2m1/float4_e1m2.
        expert_token_num_per_rank: Token counts per expert per rank of shape (N, E),
            dtype int32 or int64.
        per_token_scales: Optional scale tensor. Shape (A) for float32,
            (A, S) for float32, (A, K/64, 2) for float8_e8m0 with FP8 tokens.
        expert_topk_weight: Optional topk weight tensor of shape (A, 1), dtype float32.
            When provided, permute_topk_weight output will be produced.
        expert_token_num_type: Type of expert_token_num output (0=cumsum, 1=count).
            Default: 1 (count).
        idx_type: Index type (0=gather, 1=scatter). Default: 0 (gather).
            Scatter index only supported on Ascend 950.
        tokens_dtype: Optional override for actual data type of tokens when stored
            as uint8. Use torch_npu.hifloat8, torch_npu.float4_e2m1fn_x2,
            or torch_npu.float4_e1m2fn_x2. Default: None.

    Returns:
        A tuple of (permute_tokens, permute_per_token_scales, permute_token_idx,
                    expert_token_num, permute_topk_weight).
    """
    return torch.ops.cann_ops_transformer.moe_re_routing(
        tokens,
        expert_token_num_per_rank,
        per_token_scales=per_token_scales,
        expert_topk_weight=expert_topk_weight,
        expert_token_num_type=expert_token_num_type,
        idx_type=idx_type,
        tokens_dtype=tokens_dtype,
    )
