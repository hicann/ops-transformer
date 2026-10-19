# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

from typing import Tuple

import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library

_CONTEXT_BYTES = 1024
_MAX_K = 64
_MX_GROUP_SIZE = 32
_SCALE_ALIGNMENT = 2
_TOKEN_DTYPE_FP16 = 0
_NONQUANT_SCALE_RANK = 2
_MXFP_SCALE_RANK = 3
_FIRST_MX_DTYPE = 2
_LAST_TOKEN_DTYPE = 4


def _validate_inputs(
    schedule_context: torch.Tensor,
    expert_scales: torch.Tensor,
    layer_id: torch.Tensor,
    hidden_size: int,
    token_dtype: int,
    need_schedule: int,
) -> None:
    """Validate metadata for both eager execution and Meta/FakeTensor inference."""
    torch._check(
        _TOKEN_DTYPE_FP16 <= token_dtype <= _LAST_TOKEN_DTYPE,
        lambda: f"token_dtype must be in [0, 4], got {token_dtype}",
    )
    torch._check(
        need_schedule in (0, 1),
        lambda: f"need_schedule must be 0 or 1, got {need_schedule}",
    )
    torch._check(
        hidden_size > 0, lambda: f"hidden_size must be positive, got {hidden_size}"
    )
    torch._check(
        schedule_context.dtype == torch.int8 and schedule_context.dim() == 1,
        lambda: f"schedule_context must be 1D int8, got dtype={schedule_context.dtype}, "
        f"shape={tuple(schedule_context.shape)}",
    )
    torch._check(
        schedule_context.numel() == _CONTEXT_BYTES,
        lambda: f"schedule_context must have {_CONTEXT_BYTES} bytes, got {schedule_context.numel()}",
    )
    torch._check(
        layer_id.dtype == torch.int32 and layer_id.dim() == 1,
        lambda: f"layer_id must be 1D int32, got dtype={layer_id.dtype}, shape={tuple(layer_id.shape)}",
    )
    torch._check(
        layer_id.numel() == 1,
        lambda: f"layer_id must have shape (1,), got {tuple(layer_id.shape)}",
    )
    for name, tensor in (
        ("schedule_context", schedule_context),
        ("expert_scales", expert_scales),
        ("layer_id", layer_id),
    ):
        torch._check(
            tensor.is_contiguous(),
            lambda name=name,
            tensor=tensor: f"{name} must be contiguous, got strides={tensor.stride()}",
        )
        torch._check(
            tensor.device == schedule_context.device,
            lambda name=name,
            tensor=tensor: f"{name} must be on {schedule_context.device}, got {tensor.device}",
        )
    is_mxfp = token_dtype >= _FIRST_MX_DTYPE
    expected_rank = _MXFP_SCALE_RANK if is_mxfp else _NONQUANT_SCALE_RANK
    torch._check(
        expert_scales.dim() == expected_rank,
        lambda: f"expert_scales rank must be {expected_rank} for token_dtype={token_dtype}, "
        f"got shape={tuple(expert_scales.shape)}",
    )
    torch._check(
        expert_scales.shape[0] > 0,
        lambda: f"R must be positive, got {expert_scales.shape[0]}",
    )
    k = expert_scales.shape[1] - (1 if is_mxfp else 0)
    torch._check(k > 0, lambda: f"K must be positive, got {k}")
    torch._check(k <= _MAX_K, lambda: f"K must not exceed {_MAX_K}, got {k}")
    if is_mxfp:
        torch._check(
            expert_scales.dtype
            in (torch.uint8, getattr(torch, "float8_e8m0fnu", None)),
            lambda: f"MXFP scales must be uint8 E8M0 encodings or float8_e8m0fnu, got {expert_scales.dtype}",
        )
        groups = (hidden_size + _MX_GROUP_SIZE - 1) // _MX_GROUP_SIZE
        aligned_groups = (
            (groups + _SCALE_ALIGNMENT - 1) // _SCALE_ALIGNMENT * _SCALE_ALIGNMENT
        )
        torch._check(
            expert_scales.shape[2] == aligned_groups,
            lambda: f"MXFP scale S must be {aligned_groups} (ceil(H/32) rounded up to even) "
            f"for H={hidden_size}, got {expert_scales.shape[2]}",
        )
    else:
        torch._check(
            expert_scales.dtype == torch.float32,
            lambda: f"nonquant scales must be float32, got {expert_scales.dtype}",
        )


class _AttentionWorkerCombineOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("attention_worker_combine", category="attention")

    def sources(self):
        return ["csrc/attention/attention_worker_combine.cpp"]

    def schema(self):
        return (
            "attention_worker_combine(Tensor(a!) schedule_context, Tensor expert_scales, "
            "Tensor layer_id, int hidden_size, *, int token_dtype=0, int need_schedule=0) -> (Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def attention_worker_combine_meta(
            schedule_context,
            expert_scales,
            layer_id,
            hidden_size,
            *,
            token_dtype=0,
            need_schedule=0,
        ):
            _validate_inputs(
                schedule_context,
                expert_scales,
                layer_id,
                hidden_size,
                token_dtype,
                need_schedule,
            )
            dtype = (
                torch.float16 if token_dtype == _TOKEN_DTYPE_FP16 else torch.bfloat16
            )
            return (
                schedule_context.new_empty(
                    (expert_scales.shape[0], hidden_size), dtype=dtype
                ),
                layer_id.new_empty((1,)),
            )


_builder = _AttentionWorkerCombineOpBuilder()
_builder._ensure_initialized()


@impl(get_as_library(), _builder.name, "PrivateUse1")
def _attention_worker_combine(
    schedule_context,
    expert_scales,
    layer_id,
    hidden_size,
    *,
    token_dtype=0,
    need_schedule=0,
):
    _validate_inputs(
        schedule_context,
        expert_scales,
        layer_id,
        hidden_size,
        token_dtype,
        need_schedule,
    )
    return _builder.load().attention_worker_combine(
        schedule_context,
        expert_scales,
        layer_id,
        hidden_size,
        token_dtype,
        need_schedule,
    )


def attention_worker_combine(
    schedule_context: torch.Tensor,
    expert_scales: torch.Tensor,
    layer_id: torch.Tensor,
    hidden_size: int,
    *,
    token_dtype: int = 0,
    need_schedule: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """融合专家token，支持FP16/BF16及Ascend950的MXFP反量化。

    Args:
        schedule_context: 连续NPU int8 Tensor，shape (1024,)，包含有效ScheduleContext。
            调用方须保持其引用的token/flags缓冲存活并完成生产者同步。
        expert_scales: 非量化为float32 (R,K)；MXFP为uint8承载的E8M0编码或
            float8_e8m0fnu，shape (R,K+1,S)，S为ceil(H/32)向上取整到偶数。
            R>0、1<=K<=64，K不含共享专家。
        layer_id: 连续NPU int32 Tensor，shape (1,)。
        hidden_size: 每行逻辑元素数H，必须为正。
        token_dtype: 0 FP16、1 BF16、2 E5M2、3 E4M3FN、4 E2M1，默认0。
        need_schedule: 0读取当前微批次；1等待下一微批次就绪，消费flags并更新context。
            默认0。重复调度调用前，生产者须准备相应批次的数据和就绪标志。

    Returns:
        y为(R,H)，token_dtype=0时为float16，其余为bfloat16；
        next_layer_id为int32 (1,)，值为layer_id+1。不支持反向。
    """
    return torch.ops.cann_ops_transformer.attention_worker_combine(
        schedule_context,
        expert_scales,
        layer_id,
        hidden_size,
        token_dtype=token_dtype,
        need_schedule=need_schedule,
    )
