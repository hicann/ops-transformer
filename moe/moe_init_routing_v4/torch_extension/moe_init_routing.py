# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional, Tuple, List, Union
import math
import torch
import torch_npu
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error
from cann_ops_transformer.op_builder import OpBuilder, get_as_library

ALIGN_BASE = 64
THREE_DIM_NUM = 3
MXFPX_SCALE_THIRD_DIM_SIZE = 2
INT4_IN_UINT8 = 2
MXQUANT_BLOCK_SIZE = 32
PAD_TO_EVEN_FACTOR = 2
FP8_QUANT_BLOCK_SIZE = 128


class _MoeInitRoutingOpBuilder(OpBuilder):
    def __init__(self):
        super(_MoeInitRoutingOpBuilder, self).__init__(
            "moe_init_routing", category="moe"
        )

    @staticmethod
    def _validate_scale_no_quant(x, scale, is_fp8_or_fp4_dtype, is_float4_case):
        scale_dim = scale.dim()
        if is_fp8_or_fp4_dtype:
            torch._check(
                scale_dim == THREE_DIM_NUM,
                lambda: f"the scale shape support only 3D in no quant mode and x type is "
                f"float8_e5m2, float8_e4m3fn or float4_e2m1.{ops_error(ErrCode.VALUE)}",
            )
            torch._check(
                x.size(0) == scale.size(0),
                lambda: f"the first dim of scale and the first dim of x "
                f"should be the same.{ops_error(ErrCode.VALUE)}",
            )
            if is_float4_case:
                torch._check(
                    (x.size(1) * 2 + ALIGN_BASE - 1) // ALIGN_BASE == scale.size(1),
                    lambda: f"the scale and x must have compatible second dimensions "
                    f"(x aligned to 64).{ops_error(ErrCode.VALUE)}",
                )
            else:
                torch._check(
                    (x.size(1) + ALIGN_BASE - 1) // ALIGN_BASE == scale.size(1),
                    lambda: f"the scale and x must have compatible second dimensions "
                    f"(x aligned to 64).{ops_error(ErrCode.VALUE)}",
                )
            torch._check(
                MXFPX_SCALE_THIRD_DIM_SIZE == scale.size(2),
                lambda: f"the third dim of scale should be 2.{ops_error(ErrCode.VALUE)}",
            )
        else:
            torch._check(
                scale_dim == 1,
                lambda: f"the scale shape support only 1D in no quant mode.{ops_error(ErrCode.VALUE)}",
            )
            torch._check(
                x.size(0) == scale.size(0),
                lambda: f"the first dim of scale and the first dim of x "
                f"should be the same.{ops_error(ErrCode.VALUE)}",
            )

    @staticmethod
    def _validate_scale_static_quant(scale, offset):
        torch._check(
            scale.dim() == 1,
            lambda: f"the scale shape support only 1D in static quant mode.{ops_error(ErrCode.VALUE)}",
        )
        torch._check(
            scale.size(0) == 1,
            lambda: f"the shape of scale should be 1.{ops_error(ErrCode.VALUE)}",
        )
        if offset is not None:
            torch._check(
                offset.dim() == 1,
                lambda: f"the offset shape support only 1D.{ops_error(ErrCode.VALUE)}",
            )
            torch._check(
                scale.size(0) == offset.size(0),
                lambda: f"the 1st dim of offset and the 1st dim of scale "
                f"should be the same.{ops_error(ErrCode.VALUE)}",
            )

    @staticmethod
    def _validate_scale_dynamic_quant(x, scale, expert_range_length):
        torch._check(
            scale.dim() == 2,
            lambda: f"the scale shape support only 2D in dynamic quant mode.{ops_error(ErrCode.VALUE)}",
        )
        torch._check(
            scale.size(0) in [expert_range_length, 1],
            lambda: f"the first dim of scale must be in "
            f"[expert_range_length, 1].{ops_error(ErrCode.VALUE)}",
        )
        torch._check(
            x.size(1) == scale.size(1),
            lambda: f"the 2nd dim of scale should be the same "
            f"with the 2nd dim of x.{ops_error(ErrCode.VALUE)}",
        )

    @staticmethod
    def _validate_scale_int4_dynamic_quant(x, scale):
        torch._check(
            scale.dim() == 2,
            lambda: f"the scale shape support only 2D in "
            f"INT4 dynamic quant mode.{ops_error(ErrCode.VALUE)}",
        )
        torch._check(
            scale.size(0) == 1,
            lambda: f"the first dim of scale must be 1 in "
            f"INT4 dynamic quant mode.{ops_error(ErrCode.VALUE)}",
        )
        torch._check(
            scale.dtype == torch.float32,
            lambda: f"the scale dtype must be float32 in "
            f"INT4 dynamic quant mode.{ops_error(ErrCode.TYPE)}",
        )
        torch._check(
            x.size(1) == scale.size(1),
            lambda: f"the 2nd dim of scale should be the same "
            f"with the 2nd dim of x.{ops_error(ErrCode.VALUE)}",
        )

    @staticmethod
    def _validate_scale_hif8_pertensor(scale):
        torch._check(
            scale.dim() == 1,
            lambda: f"the scale shape support only 1D in "
            f"HIF8 pertensor quant mode.{ops_error(ErrCode.VALUE)}",
        )

    @staticmethod
    def _validate_int4_dynamic_quant(x, offset, drop_pad_mode, x_dtype):
        torch._check(
            drop_pad_mode == 0,
            lambda: f"INT4 dynamic quantization only supports drop_pad_mode=0.{ops_error(ErrCode.VALUE)}",
        )
        torch._check(
            x.dtype in [torch.float32, torch.bfloat16],
            lambda: f"INT4 dynamic quantization only supports float32 or bfloat16 x.{ops_error(ErrCode.TYPE)}",
        )
        torch._check(
            offset is None,
            lambda: f"INT4 dynamic quantization does not support offset.{ops_error(ErrCode.VALUE)}",
        )
        torch._check(
            x.size(1) % INT4_IN_UINT8 == 0,
            lambda: f"INT4 dynamic quantization requires x.size(1) to be even.{ops_error(ErrCode.VALUE)}",
        )
        torch._check(
            x_dtype is None or x_dtype == torch_npu.int4,
            lambda: f"x_dtype must be torch_npu.int4 or None when quant_mode=13.{ops_error(ErrCode.VALUE)}",
        )

    @staticmethod
    def _validate_topk_weight(topk_weight, x, expert_idx):
        torch._check(
            topk_weight.dtype == torch.float32,
            lambda: f"topk_weight only supports float32, but got {topk_weight.dtype}.{ops_error(ErrCode.TYPE)}",
        )
        torch._check(
            topk_weight.dim() == 2,
            lambda: f"topk_weight must be 2D, but got {topk_weight.dim()}D.{ops_error(ErrCode.VALUE)}",
        )
        torch._check(
            topk_weight.size(0) == x.size(0),
            lambda: f"topk_weight.size(0) must equal x.size(0).{ops_error(ErrCode.VALUE)}",
        )
        torch._check(
            topk_weight.size(1) == expert_idx.size(1),
            lambda: f"topk_weight.size(1) must equal expert_idx.size(1).{ops_error(ErrCode.VALUE)}",
        )

    @staticmethod
    def _validate_active_expert_range(active_expert_range, expert_num):
        if active_expert_range is None:
            return expert_num
        torch._check(
            isinstance(active_expert_range, list) and len(active_expert_range) == 2,
            lambda: f"active_expert_range must be a list of 2 ints.{ops_error(ErrCode.VALUE)}",
        )
        torch._check(
            active_expert_range[1] > active_expert_range[0],
            lambda: f"active_expert_range must be increasing.{ops_error(ErrCode.VALUE)}",
        )
        torch._check(
            active_expert_range[0] >= 0 and active_expert_range[1] <= 10240,
            lambda: f"active_expert_range must be within [0, 10240].{ops_error(ErrCode.VALUE)}",
        )
        return active_expert_range[1] - active_expert_range[0]

    @staticmethod
    def _validate_inputs(
        x,
        expert_idx,
        scale,
        offset,
        active_num,
        expert_num,
        expert_capacity,
        drop_pad_mode,
        expert_tokens_num_type,
        quant_mode,
        active_expert_range,
        row_idx_type,
        x_dtype,
        topk_weight,
    ):
        torch._check(
            x.dim() == 2,
            lambda: f"x must be 2D, but got {x.dim()}D.{ops_error(ErrCode.VALUE)}",
        )
        torch._check(
            expert_idx.dim() == 2,
            lambda: f"expert_idx must be 2D, but got {expert_idx.dim()}D.{ops_error(ErrCode.VALUE)}",
        )
        torch._check(
            x.size(0) == expert_idx.size(0),
            lambda: f"the first dim of expert_idx and x should be the same.{ops_error(ErrCode.VALUE)}",
        )
        torch._check(
            expert_num > 0,
            lambda: f"expert_num must be greater than 0.{ops_error(ErrCode.VALUE)}",
        )
        expert_range_length = _MoeInitRoutingOpBuilder._validate_active_expert_range(
            active_expert_range, expert_num
        )
        torch._check(
            drop_pad_mode in [0, 1],
            lambda: f"drop_pad_mode must be 0 or 1.{ops_error(ErrCode.VALUE)}",
        )
        if drop_pad_mode == 1:
            torch._check(
                expert_capacity > 0,
                lambda: f"expert_capacity must be greater than 0 when drop_pad_mode=1.{ops_error(ErrCode.VALUE)}",
            )
        torch._check(
            expert_tokens_num_type in [0, 1, 2],
            lambda: f"expert_tokens_num_type must be 0, 1 or 2.{ops_error(ErrCode.VALUE)}",
        )
        torch._check(
            quant_mode
            in [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17],
            lambda: f"quant_mode must be in [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, "
            f"14, 15, 16, 17].{ops_error(ErrCode.VALUE)}",
        )
        torch._check(
            row_idx_type in [0, 1],
            lambda: f"row_idx_type must be 0 or 1.{ops_error(ErrCode.VALUE)}",
        )
        is_float4_case = (
            hasattr(torch, "float4_e2m1fn_x2") and x.dtype == torch.float4_e2m1fn_x2
        ) or x_dtype == torch_npu.float4_e2m1fn_x2
        is_fp8_or_fp4_dtype = (
            x.dtype == torch.float8_e5m2
            or x.dtype == torch.float8_e4m3fn
            or is_float4_case
        )
        is_dynamic_int4_output = quant_mode == 13
        torch._check(
            not (quant_mode == 1 and x_dtype == torch_npu.int4),
            lambda: f"INT4 dynamic quantization uses quant_mode=13. "
            f"quant_mode=1 only supports INT8 dynamic quantization.{ops_error(ErrCode.VALUE)}",
        )
        if is_dynamic_int4_output:
            _MoeInitRoutingOpBuilder._validate_int4_dynamic_quant(
                x, offset, drop_pad_mode, x_dtype
            )
        if scale is not None:
            _MoeInitRoutingOpBuilder._validate_scale_params(
                x,
                scale,
                offset,
                quant_mode,
                expert_range_length,
                is_fp8_or_fp4_dtype,
                is_float4_case,
            )
        if topk_weight is not None:
            _MoeInitRoutingOpBuilder._validate_topk_weight(topk_weight, x, expert_idx)
        return (
            is_float4_case,
            is_fp8_or_fp4_dtype,
            is_dynamic_int4_output,
            expert_range_length,
        )

    @staticmethod
    def _validate_scale_params(
        x,
        scale,
        offset,
        quant_mode,
        expert_range_length,
        is_fp8_or_fp4_dtype,
        is_float4_case,
    ):
        if quant_mode == -1:
            _MoeInitRoutingOpBuilder._validate_scale_no_quant(
                x, scale, is_fp8_or_fp4_dtype, is_float4_case
            )
        elif quant_mode == 0:
            _MoeInitRoutingOpBuilder._validate_scale_static_quant(scale, offset)
        elif quant_mode == 1:
            _MoeInitRoutingOpBuilder._validate_scale_dynamic_quant(
                x, scale, expert_range_length
            )
        elif quant_mode == 13:
            _MoeInitRoutingOpBuilder._validate_scale_int4_dynamic_quant(x, scale)
        if quant_mode == 7:
            _MoeInitRoutingOpBuilder._validate_scale_hif8_pertensor(scale)

    @staticmethod
    def _compute_output_dtypes(
        x, x_dtype, quant_mode, is_fp8_or_fp4_dtype, is_dynamic_int4_output
    ):
        expanded_x_dtype = x.dtype
        expanded_scale_dtype = torch.float32
        if x_dtype == torch_npu.hifloat8:
            expanded_x_dtype = torch.uint8
        if is_dynamic_int4_output:
            expanded_x_dtype = torch.uint8
        elif quant_mode in [0, 1]:
            expanded_x_dtype = torch.int8
        elif quant_mode in [2, 16]:
            expanded_x_dtype = torch.float8_e5m2
            expanded_scale_dtype = torch.float8_e8m0fnu
        elif quant_mode in [3, 17]:
            expanded_x_dtype = torch.float8_e4m3fn
            expanded_scale_dtype = torch.float8_e8m0fnu
        elif quant_mode in [6, 7, 8]:
            expanded_x_dtype = torch.uint8
        elif quant_mode == -1 and is_fp8_or_fp4_dtype:
            expanded_scale_dtype = torch.float8_e8m0fnu
        elif quant_mode == 9:
            expanded_x_dtype = torch.uint8
            expanded_scale_dtype = torch.float8_e8m0fnu
        elif quant_mode in [4, 11, 14]:
            expanded_x_dtype = torch.float8_e5m2
        elif quant_mode in [5, 12, 15]:
            expanded_x_dtype = torch.float8_e4m3fn
        return expanded_x_dtype, expanded_scale_dtype

    @staticmethod
    def _compute_output_dims(
        x,
        h,
        bs,
        k,
        expert_num,
        expert_capacity,
        active_num,
        drop_pad_mode,
        quant_mode,
        is_fp8_or_fp4_dtype,
        is_dynamic_int4_output,
        scale,
    ):
        expanded_x_dim_list = []
        expanded_scale_dim_list = []

        if drop_pad_mode == 1:
            expanded_x_dim_list = [expert_num, expert_capacity, h]
            expanded_scale_dim_list = [expert_num * expert_capacity]
        else:
            num_expanded_rows = bs * k
            expanded_x_dim_list = [num_expanded_rows, x.size(1)]
            if quant_mode in [2, 3, 16, 17]:
                scale_cols = (h + MXQUANT_BLOCK_SIZE - 1) // MXQUANT_BLOCK_SIZE
                scale_cols = (
                    (scale_cols + PAD_TO_EVEN_FACTOR - 1)
                    // PAD_TO_EVEN_FACTOR
                    * PAD_TO_EVEN_FACTOR
                )
                expanded_scale_dim_list = [num_expanded_rows, scale_cols]
            elif quant_mode == -1 and is_fp8_or_fp4_dtype and scale is not None:
                scale_cols = (h + ALIGN_BASE - 1) // ALIGN_BASE
                expanded_scale_dim_list = [num_expanded_rows, scale_cols, 2]
            elif is_dynamic_int4_output:
                expanded_x_dim_list = [num_expanded_rows, x.size(1) // INT4_IN_UINT8]
                expanded_scale_dim_list = [num_expanded_rows]
            elif quant_mode in [-1, 1, 8]:
                expanded_scale_dim_list = [num_expanded_rows]
            elif quant_mode == 9:
                expanded_x_dim_list = [num_expanded_rows, x.size(1) // 2]
                scale_cols = math.ceil(h / ALIGN_BASE)
                expanded_scale_dim_list = [num_expanded_rows, scale_cols, 2]
            elif quant_mode in [11, 12]:
                block_size = FP8_QUANT_BLOCK_SIZE * PAD_TO_EVEN_FACTOR
                scale_cols = math.ceil(h / block_size)
                expanded_scale_dim_list = [
                    num_expanded_rows,
                    scale_cols,
                    PAD_TO_EVEN_FACTOR,
                ]
            elif quant_mode in [4, 5, 14, 15]:
                expanded_scale_dim_list = [
                    num_expanded_rows,
                    math.ceil(h / FP8_QUANT_BLOCK_SIZE),
                ]

        if quant_mode in [0, 6, 7]:
            expanded_scale_dim_list = []

        return expanded_x_dim_list, expanded_scale_dim_list

    def sources(self):
        return ["csrc/moe/moe_init_routing.cpp"]

    def schema(self) -> str:
        return (
            "moe_init_routing(Tensor x, Tensor expert_idx, "
            "*, Tensor? scale=None, Tensor? offset=None, Tensor? topk_weight=None, "
            "SymInt active_num=-1, int expert_capacity=-1, int expert_num=-1, "
            "int drop_pad_mode=0, int expert_tokens_num_type=0, bool expert_tokens_num_flag=False, "
            "int quant_mode=-1, int[]? active_expert_range=None, int row_idx_type=0, "
            "int? x_dtype=None) "
            "-> (Tensor, Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def moe_init_routing_meta(
            x,
            expert_idx,
            scale=None,
            offset=None,
            topk_weight=None,
            active_num=-1,
            expert_capacity=-1,
            expert_num=-1,
            drop_pad_mode=0,
            expert_tokens_num_type=0,
            expert_tokens_num_flag=False,
            quant_mode=-1,
            active_expert_range=None,
            row_idx_type=0,
            x_dtype=None,
        ):
            (
                is_float4_case,
                is_fp8_or_fp4_dtype,
                is_dynamic_int4_output,
                expert_range_length,
            ) = _MoeInitRoutingOpBuilder._validate_inputs(
                x,
                expert_idx,
                scale,
                offset,
                active_num,
                expert_num,
                expert_capacity,
                drop_pad_mode,
                expert_tokens_num_type,
                quant_mode,
                active_expert_range,
                row_idx_type,
                x_dtype,
                topk_weight,
            )
            bs = x.size(0)
            h = x.size(1)
            if is_float4_case:
                h = x.size(1) * 2
            k = expert_idx.size(1)

            expanded_x_dtype, expanded_scale_dtype = (
                _MoeInitRoutingOpBuilder._compute_output_dtypes(
                    x, x_dtype, quant_mode, is_fp8_or_fp4_dtype, is_dynamic_int4_output
                )
            )

            expanded_x_dim_list, expanded_scale_dim_list = (
                _MoeInitRoutingOpBuilder._compute_output_dims(
                    x,
                    h,
                    bs,
                    k,
                    expert_num,
                    expert_capacity,
                    active_num,
                    drop_pad_mode,
                    quant_mode,
                    is_fp8_or_fp4_dtype,
                    is_dynamic_int4_output,
                    scale,
                )
            )

            expanded_row_idx_dim_list = [bs * k]

            if not expert_tokens_num_flag:
                expert_token_cumsum_or_count_dim_list = []
            elif expert_tokens_num_type < 2:
                expert_token_cumsum_or_count_dim_list = [expert_range_length]
            else:
                expert_token_cumsum_or_count_dim_list = [expert_num, 2]

            if topk_weight is not None:
                if drop_pad_mode == 1:
                    topk_weight_rows = expert_num * expert_capacity
                else:
                    topk_weight_rows = bs * k
                expanded_topk_weight = x.new_empty(
                    (topk_weight_rows, 1), dtype=torch.float32
                )
            else:
                expanded_topk_weight = x.new_empty((0,), dtype=torch.float32)

            return (
                x.new_empty(tuple(expanded_x_dim_list), dtype=expanded_x_dtype),
                x.new_empty(tuple(expanded_row_idx_dim_list), dtype=torch.int32),
                x.new_empty(
                    tuple(expert_token_cumsum_or_count_dim_list), dtype=torch.int64
                ),
                x.new_empty(tuple(expanded_scale_dim_list), dtype=expanded_scale_dtype),
                expanded_topk_weight,
            )


_moe_init_routing_op_builder = _MoeInitRoutingOpBuilder()
_moe_init_routing_op_builder._ensure_initialized()


@impl(get_as_library(), _moe_init_routing_op_builder.name, "PrivateUse1")
def _moe_init_routing(
    x,
    expert_idx,
    scale=None,
    offset=None,
    topk_weight=None,
    active_num=-1,
    expert_capacity=-1,
    expert_num=-1,
    drop_pad_mode=0,
    expert_tokens_num_type=0,
    expert_tokens_num_flag=False,
    quant_mode=-1,
    active_expert_range=None,
    row_idx_type=0,
    x_dtype=None,
):
    op_module = _moe_init_routing_op_builder.load()
    return op_module.moe_init_routing(
        x,
        expert_idx,
        scale,
        offset,
        topk_weight,
        active_num,
        expert_capacity,
        expert_num,
        drop_pad_mode,
        expert_tokens_num_type,
        expert_tokens_num_flag,
        quant_mode,
        active_expert_range,
        row_idx_type,
        x_dtype,
    )


def _has_value(t):
    if t is None:
        return False
    if isinstance(t, torch.Tensor):
        return t.numel() > 0
    return True


_NUM_FORWARD_INPUTS = 15


def _check_backward_supported(ctx):
    """Raise NotImplementedError if V4-specific features that V2Grad cannot handle are used."""
    if ctx.has_v4_extras:
        raise NotImplementedError(
            "moe_init_routing autograd is not supported when V4-specific "
            "inputs (scale, offset, topk_weight) are provided or quant_mode "
            "is not -1 (no quantization), row_idx_type is not 0, or x_dtype "
            "is set. The aclnnMoeInitRoutingV2Grad op only handles the "
            "unquantized routing case."
        )
    if ctx.drop_pad_mode not in (0, 1):
        raise NotImplementedError(
            "moe_init_routing autograd is only supported for drop_pad_mode "
            "0 or 1. The backward op (aclnnMoeInitRoutingV2Grad) does not "
            f"support drop_pad_mode {ctx.drop_pad_mode}."
        )


class MoeInitRoutingFn(torch.autograd.Function):
    """Autograd binding: forward -> aclnnMoeInitRoutingV4,
    backward -> aclnnMoeInitRoutingV2Grad.

    The V2Grad op only handles the unquantized routing case.  Autograd is
    therefore only supported when V4 reduces to V2 — i.e. none of the
    V4-specific inputs (``scale``, ``offset``, ``topk_weight``) are provided,
    ``quant_mode`` is -1 (no quantization), ``row_idx_type`` is 0, and
    ``x_dtype`` is None.  ``active_expert_range`` does not affect the backward
    (the V2Grad op does not consume it) and is therefore not treated as a
    V4-specific feature.  When V4-specific features are used, backward raises
    ``NotImplementedError``.

    Only ``expanded_x`` is differentiable (it is a permuted copy of ``x``).
    ``expanded_row_idx`` and the other integer/token-count outputs have no
    gradient.  ``expert_idx`` is an integer index tensor with no gradient.

    The grad op computes ``grad_x`` by scattering ``grad_expanded_x`` back to
    the original rows of ``x`` according to ``expanded_row_idx`` and ``topK``.
    """

    @staticmethod
    def forward(
        ctx,
        x,
        expert_idx,
        scale,
        offset,
        topk_weight,
        active_num,
        expert_capacity,
        expert_num,
        drop_pad_mode,
        expert_tokens_num_type,
        expert_tokens_num_flag,
        quant_mode,
        active_expert_range,
        row_idx_type,
        x_dtype,
    ):
        ctx.drop_pad_mode = drop_pad_mode if drop_pad_mode is not None else 0
        ctx.k = expert_idx.size(1)
        ctx.active_num = active_num if active_num is not None else -1
        ctx.has_v4_extras = (
            _has_value(scale)
            or _has_value(offset)
            or _has_value(topk_weight)
            or quant_mode != -1
            or row_idx_type != 0
            or x_dtype is not None
        )
        (
            expanded_x,
            expanded_row_idx,
            expert_tokens_count_or_cumsum,
            expanded_scale,
            expanded_topk_weight,
        ) = torch.ops.cann_ops_transformer.moe_init_routing(
            x,
            expert_idx,
            scale=scale,
            offset=offset,
            topk_weight=topk_weight,
            active_num=active_num,
            expert_capacity=expert_capacity,
            expert_num=expert_num,
            drop_pad_mode=drop_pad_mode,
            expert_tokens_num_type=expert_tokens_num_type,
            expert_tokens_num_flag=expert_tokens_num_flag,
            quant_mode=quant_mode,
            active_expert_range=active_expert_range,
            row_idx_type=row_idx_type,
            x_dtype=x_dtype,
        )
        ctx.save_for_backward(expanded_row_idx)
        return (
            expanded_x,
            expanded_row_idx,
            expert_tokens_count_or_cumsum,
            expanded_scale,
            expanded_topk_weight,
        )

    @staticmethod
    def backward(
        ctx,
        grad_expanded_x,
        grad_expanded_row_idx,
        grad_expert_tokens_count_or_cumsum,
        grad_expanded_scale,
        grad_expanded_topk_weight,
    ):
        _check_backward_supported(ctx)
        needs = ctx.needs_input_grad
        if grad_expanded_x is None or not needs[0]:
            return (None,) * _NUM_FORWARD_INPUTS

        (expanded_row_idx,) = ctx.saved_tensors
        grad_expanded_x = grad_expanded_x.contiguous()
        active_num = ctx.active_num
        if isinstance(active_num, torch.SymInt):
            try:
                active_num = int(active_num)
            except Exception:
                active_num = 0
        if active_num < 0:
            active_num = 0
        grad_x = torch.ops.cann_ops_transformer.moe_init_routing_grad(
            grad_expanded_x, expanded_row_idx, ctx.k, ctx.drop_pad_mode, active_num
        )
        grad_x = grad_x if needs[0] else None
        return (
            grad_x,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def moe_init_routing(
    x: torch.Tensor,
    expert_idx: torch.Tensor,
    scale: Optional[torch.Tensor] = None,
    offset: Optional[torch.Tensor] = None,
    topk_weight: Optional[torch.Tensor] = None,
    active_num: Union[int, torch.SymInt] = -1,
    expert_capacity: int = -1,
    expert_num: int = -1,
    drop_pad_mode: int = 0,
    expert_tokens_num_type: int = 0,
    expert_tokens_num_flag: bool = False,
    quant_mode: int = -1,
    active_expert_range: Optional[List[int]] = None,
    row_idx_type: int = 0,
    x_dtype: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """moe_init_routing operator.

    Performs MoE (Mixture of Experts) init routing, expanding tokens to their
    assigned experts and producing routing indices.

    Supports automatic differentiation for ``x``: when ``x.requires_grad`` is
    True and V4-specific features (quantization, scale, offset, topk_weight,
    etc.) are not used, the backward pass calls
    :func:`moe_init_routing_grad` (wrapping ``aclnnMoeInitRoutingV2Grad``) to
    scatter ``grad_expanded_x`` back to ``grad_x``.  When V4-specific features
    are used, backward raises ``NotImplementedError``.

    Args:
        x (Tensor): Input tensor of shape (n, cols), dtype supports float16/bfloat16/float32/int8/float8.
        expert_idx (Tensor): Expert indices of shape (n, k), dtype int32. n must equal x.size(0).
        scale (Tensor, optional): Scale tensor for quantization.
            - quant_mode=-1 & non-fp8/fp4: shape (n,), dtype float32.
            - quant_mode=-1 & fp8/fp4: shape (n, ceil(cols_align64), 2), dtype float8_e8m0fnu.
            - quant_mode=0: shape (1,), dtype float32.
            - quant_mode=1: shape (expert_range_length or 1, cols), dtype float32.
            - quant_mode=2/3: not used.
            - quant_mode=7: shape (n,), dtype float32.
            - quant_mode=13: shape (1, cols), dtype float32.
            Default: None.
        offset (Tensor, optional): Offset tensor for quantization.
            - quant_mode=0: shape same as scale, dtype float32.
            - Other modes: not used. Default: None.
        topk_weight (Tensor, optional): Topk weight tensor of shape (n, k), dtype float32.
            When provided, expanded_topk_weight output will also be produced. Default: None.
        active_num (Union[int, torch.SymInt]): Number of active tokens. -1 means n*k (all tokens).
            Supports SymInt for dynamic shape in graph mode. Default: -1.
        expert_capacity (int): Capacity per expert, required when drop_pad_mode=1. Default: -1.
        expert_num (int): Total number of experts, must be greater than 0. Default: -1.
        drop_pad_mode (int): 0 for dropless, 1 for drop/pad mode. Default: 0.
        expert_tokens_num_type (int): Type of expert token count output.
            0=count, 1=cumsum, 2=key_value. Default: 0.
        expert_tokens_num_flag (bool): Whether to produce expert_tokens_count output. Default: False.
        quant_mode (int): Quantization mode. Default: -1.
            -1: no quantization; 0: static; 1: dynamic int8; 2: mxfp8_e5m2;
            3: mxfp8_e4m3fn; 4: fp8_group_e5m2; 5: fp8_group_e4m3fn;
            6: hif8_cast; 7: hif8_pertensor; 8: hif8_per_token_dim;
            9: mxfp4_e2m1; 11: fp8_perblock_e5m2; 12: fp8_perblock_e4m3fn;
            13: int4_dynamic; 14: fp8_group_amax_e5m2; 15: fp8_group_amax_e4m3fn;
            16: mxfp8_roundscale_amax_e5m2; 17: mxfp8_roundscale_amax_e4m3fn.
        active_expert_range (list[int], optional): Range of active experts [start, end].
            start >= 0, end <= 10240, end > start. When None, defaults to
            ``[0, expert_num]`` (all experts active). Default: None.
        row_idx_type (int): Row index type. 0=gather, 1=scatter. Default: 0.
        x_dtype (int, optional): Override for actual data type of x when x is stored as uint8
            but represents a different type. Use torch_npu.float4_e2m1fn_x2,
            torch_npu.hifloat8, or torch_npu.int4. Default: None.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
            - expanded_x (Tensor): Expanded input tensor.
                - drop_pad_mode=0: shape (active_num, cols), dtype depends on quant_mode.
                - drop_pad_mode=1: shape (expert_num, expert_capacity, cols), dtype depends on quant_mode.
            - expanded_row_idx (Tensor): Row indices of shape (n*k,), dtype int32.
            - expert_tokens_count_or_cumsum (Tensor): Expert token counts.
                - expert_tokens_num_flag=False: empty tensor, dtype int64.
                - expert_tokens_num_type<2: shape (expert_range_length,), dtype int64.
                - expert_tokens_num_type=2: shape (expert_num, 2), dtype int64.
            - expanded_scale (Tensor): Expanded scale tensor.
                - Shape and dtype depend on quant_mode (see scale for details).
                - quant_mode=0/6/7: empty tensor.
            - expanded_topk_weight (Tensor): Expanded topk weights.
                - topk_weight provided: shape (active_num, 1) or (expert_num*expert_capacity, 1), dtype float32.
                - topk_weight not provided: shape (0,), dtype float32.
    """
    needs_grad = x.requires_grad
    if active_expert_range is None:
        active_expert_range = [0, expert_num]
    if needs_grad:
        return MoeInitRoutingFn.apply(
            x,
            expert_idx,
            scale,
            offset,
            topk_weight,
            active_num,
            expert_capacity,
            expert_num,
            drop_pad_mode,
            expert_tokens_num_type,
            expert_tokens_num_flag,
            quant_mode,
            active_expert_range,
            row_idx_type,
            x_dtype,
        )
    return torch.ops.cann_ops_transformer.moe_init_routing(
        x,
        expert_idx,
        scale=scale,
        offset=offset,
        topk_weight=topk_weight,
        active_num=active_num,
        expert_capacity=expert_capacity,
        expert_num=expert_num,
        drop_pad_mode=drop_pad_mode,
        expert_tokens_num_type=expert_tokens_num_type,
        expert_tokens_num_flag=expert_tokens_num_flag,
        quant_mode=quant_mode,
        active_expert_range=active_expert_range,
        row_idx_type=row_idx_type,
        x_dtype=x_dtype,
    )
