# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import math
from typing import Optional

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library

MATRIX_DIM_NUM = 2
VECTOR_DIM_NUM = 1
TOKEN_DIM_INDEX = 0
HIDDEN_DIM_INDEX = 1
MAX_HIDDEN_SIZE = 8192
DEFAULT_EPS = 1.0e-6
MIN_OUTPUT_NORM_EPS = torch.finfo(torch.float32).tiny
MAX_OUTPUT_NORM_EPS = torch.finfo(torch.float32).max


class BlockAttnResUpdateOpBuilder(OpBuilder):
    def __init__(self):
        super(BlockAttnResUpdateOpBuilder, self).__init__(
            "block_attn_res_update", category="attention"
        )

    def sources(self):
        return ["csrc/attention/block_attn_res_update.cpp"]

    def schema(self) -> str:
        return (
            "block_attn_res_update("
            "Tensor(a!) partial_block, Tensor delta, Tensor pseudo_query, "
            "Tensor numerator, Tensor logit_max, Tensor exp_sum, *, "
            f"float eps={DEFAULT_EPS}, Tensor? output_norm_weight=None, "
            f"float output_norm_eps={DEFAULT_EPS}) -> Tensor"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def block_attn_res_update_meta(
            partial_block,
            delta,
            pseudo_query,
            numerator,
            logit_max,
            exp_sum,
            *,
            eps=DEFAULT_EPS,
            output_norm_weight=None,
            output_norm_eps=DEFAULT_EPS,
        ):
            torch._check(
                partial_block.dtype == torch.float32,
                lambda: f"partial_block must be float32, but got {partial_block.dtype}.",
            )
            torch._check(
                delta.dtype == torch.bfloat16,
                lambda: f"delta must be bfloat16, but got {delta.dtype}.",
            )
            for name, tensor in (
                ("pseudo_query", pseudo_query),
                ("numerator", numerator),
                ("logit_max", logit_max),
                ("exp_sum", exp_sum),
            ):
                torch._check(
                    tensor.dtype == torch.float32,
                    lambda name=name, tensor=tensor: (
                        f"{name} must be float32, but got {tensor.dtype}."
                    ),
                )

            for name, tensor in (
                ("partial_block", partial_block),
                ("delta", delta),
                ("pseudo_query", pseudo_query),
                ("numerator", numerator),
                ("logit_max", logit_max),
                ("exp_sum", exp_sum),
            ):
                torch._check(
                    tensor.device == partial_block.device,
                    lambda name=name, tensor=tensor: (
                        f"{name} must be on the same device as partial_block, but got "
                        f"{tensor.device} and {partial_block.device}."
                    ),
                )
                torch._check(
                    tensor.is_contiguous(),
                    lambda name=name: f"{name} must be contiguous.",
                )

            torch._check(
                math.isfinite(eps) and eps > 0.0,
                lambda: f"eps must be finite and greater than 0, but got {eps}.",
            )

            torch._check(
                partial_block.dim() == MATRIX_DIM_NUM,
                lambda: (
                    f"partial_block must be {MATRIX_DIM_NUM}D, "
                    f"but got {partial_block.dim()}D."
                ),
            )
            torch._check(
                delta.dim() == MATRIX_DIM_NUM,
                lambda: f"delta must be {MATRIX_DIM_NUM}D, but got {delta.dim()}D.",
            )
            torch._check(
                numerator.dim() == MATRIX_DIM_NUM,
                lambda: (
                    f"numerator must be {MATRIX_DIM_NUM}D, but got {numerator.dim()}D."
                ),
            )
            torch._check(
                pseudo_query.dim() == VECTOR_DIM_NUM,
                lambda: (
                    f"pseudo_query must be {VECTOR_DIM_NUM}D, "
                    f"but got {pseudo_query.dim()}D."
                ),
            )
            torch._check(
                logit_max.dim() == VECTOR_DIM_NUM,
                lambda: (
                    f"logit_max must be {VECTOR_DIM_NUM}D, but got {logit_max.dim()}D."
                ),
            )
            torch._check(
                exp_sum.dim() == VECTOR_DIM_NUM,
                lambda: (
                    f"exp_sum must be {VECTOR_DIM_NUM}D, but got {exp_sum.dim()}D."
                ),
            )

            torch._check(
                partial_block.shape[TOKEN_DIM_INDEX] == delta.shape[TOKEN_DIM_INDEX],
                lambda: (
                    "delta token dimension must match partial_block, but got "
                    f"{delta.shape} and {partial_block.shape}."
                ),
            )
            torch._check(
                partial_block.shape[HIDDEN_DIM_INDEX] == delta.shape[HIDDEN_DIM_INDEX],
                lambda: (
                    "delta hidden dimension must match partial_block, but got "
                    f"{delta.shape} and {partial_block.shape}."
                ),
            )
            torch._check(
                partial_block.shape[TOKEN_DIM_INDEX]
                == numerator.shape[TOKEN_DIM_INDEX],
                lambda: (
                    "numerator token dimension must match partial_block, but got "
                    f"{numerator.shape} and {partial_block.shape}."
                ),
            )
            torch._check(
                partial_block.shape[HIDDEN_DIM_INDEX]
                == numerator.shape[HIDDEN_DIM_INDEX],
                lambda: (
                    "numerator hidden dimension must match partial_block, but got "
                    f"{numerator.shape} and {partial_block.shape}."
                ),
            )
            torch._check(
                pseudo_query.shape[TOKEN_DIM_INDEX]
                == partial_block.shape[HIDDEN_DIM_INDEX],
                lambda: (
                    "pseudo_query length must match the hidden dimension of "
                    f"partial_block, but got {pseudo_query.shape} and "
                    f"{partial_block.shape}."
                ),
            )
            torch._check(
                logit_max.shape[TOKEN_DIM_INDEX]
                == partial_block.shape[TOKEN_DIM_INDEX],
                lambda: (
                    "logit_max length must match the token dimension of partial_block, "
                    f"but got {logit_max.shape} and {partial_block.shape}."
                ),
            )
            torch._check(
                exp_sum.shape[TOKEN_DIM_INDEX] == partial_block.shape[TOKEN_DIM_INDEX],
                lambda: (
                    "exp_sum length must match the token dimension of "
                    f"partial_block, but got {exp_sum.shape} and "
                    f"{partial_block.shape}."
                ),
            )
            torch._check(
                partial_block.shape[TOKEN_DIM_INDEX] >= 0,
                lambda: (
                    "the token dimension T must be greater than or equal to 0, but got "
                    f"{partial_block.shape[TOKEN_DIM_INDEX]}."
                ),
            )
            torch._check(
                partial_block.shape[HIDDEN_DIM_INDEX] >= 0,
                lambda: (
                    "the hidden dimension D must be greater than or equal to 0, but got "
                    f"{partial_block.shape[HIDDEN_DIM_INDEX]}."
                ),
            )
            torch._check(
                partial_block.shape[HIDDEN_DIM_INDEX] <= MAX_HIDDEN_SIZE,
                lambda: (
                    f"the hidden dimension D must not exceed {MAX_HIDDEN_SIZE}, but got "
                    f"{partial_block.shape[HIDDEN_DIM_INDEX]}."
                ),
            )
            if output_norm_weight is not None:
                torch._check(
                    output_norm_weight.dtype in (torch.bfloat16, torch.float32),
                    lambda: f"output_norm_weight must be bfloat16 or float32, but got {output_norm_weight.dtype}.",
                )
                torch._check(
                    output_norm_weight.is_contiguous(),
                    lambda: (
                        "output_norm_weight must be contiguous, but got shape "
                        f"{output_norm_weight.shape} and stride {output_norm_weight.stride()}."
                    ),
                )
                torch._check(
                    output_norm_weight.device == partial_block.device,
                    lambda: (
                        "output_norm_weight must be on the same device as partial_block, but got "
                        f"{output_norm_weight.device} and {partial_block.device}."
                    ),
                )
                torch._check(
                    output_norm_weight.dim() == VECTOR_DIM_NUM,
                    lambda: f"output_norm_weight must be 1D, but got {output_norm_weight.dim()}D.",
                )
                torch._check(
                    output_norm_weight.shape[0]
                    == partial_block.shape[HIDDEN_DIM_INDEX],
                    lambda: (
                        "output_norm_weight must have shape [D], but got "
                        f"{output_norm_weight.shape} with D={partial_block.shape[HIDDEN_DIM_INDEX]}."
                    ),
                )
                torch._check(
                    MIN_OUTPUT_NORM_EPS <= output_norm_eps <= MAX_OUTPUT_NORM_EPS,
                    lambda: (
                        "output_norm_eps must be in [FLT_MIN, FLT_MAX] before conversion to float32, but got "
                        f"{output_norm_eps}."
                    ),
                )
            h = torch.empty(delta.shape, dtype=delta.dtype, device="meta")
            return h


_block_attn_res_update_op_builder = BlockAttnResUpdateOpBuilder()
_block_attn_res_update_op_builder._ensure_initialized()


@impl(get_as_library(), _block_attn_res_update_op_builder.name, "PrivateUse1")
def _block_attn_res_update(
    partial_block,
    delta,
    pseudo_query,
    numerator,
    logit_max,
    exp_sum,
    *,
    eps=DEFAULT_EPS,
    output_norm_weight=None,
    output_norm_eps=DEFAULT_EPS,
):
    op_module = _block_attn_res_update_op_builder.load()
    return op_module.block_attn_res_update(
        partial_block,
        delta,
        pseudo_query,
        numerator,
        logit_max,
        exp_sum,
        eps,
        output_norm_weight,
        output_norm_eps,
    )


def block_attn_res_update(
    partial_block: torch.Tensor,
    delta: torch.Tensor,
    pseudo_query: torch.Tensor,
    numerator: torch.Tensor,
    logit_max: torch.Tensor,
    exp_sum: torch.Tensor,
    *,
    eps: float = DEFAULT_EPS,
    output_norm_weight: Optional[torch.Tensor] = None,
    output_norm_eps: float = DEFAULT_EPS,
) -> torch.Tensor:
    """Update the partial residual and optionally apply RMSNorm to the BF16 mixed output.

    封装 ``block_attn_res_update`` 的 ``aclnnBlockAttnResUpdateV2`` 调用。

    将当前 ``delta`` 原地累加到 FP32 ``partial_block``，计算当前
    ``partial_block`` 的 RMSNorm score，并与 ``block_attn_res_prepare``
    产生的 online softmax 中间状态合并并返回当前层结果。更新后的
    ``partial_block`` 原地写回 ``partial_block``，不作为独立返回值返回。

    Args:
        partial_block (Tensor): 已累计的 ``partial_block``，shape ``(T, D)``，
            dtype 为 ``torch.float32``，且必须连续；``T >= 0``，
            ``0 <= D <= 8192``。
            调用完成后被原地更新。
        delta (Tensor): 当前层增量，shape ``(T, D)``，dtype 为
            ``torch.bfloat16``，且必须连续。
        pseudo_query (Tensor): 用于计算当前 ``partial_block`` logit 的
            ``pseudo_query``，shape ``(D,)``，dtype 为 ``torch.float32``，且必须连续。
        numerator (Tensor): ``block_attn_res_prepare`` 输出的历史 online softmax 加权和，
            shape ``(T, D)``，dtype 为 ``torch.float32``，且必须连续。
        logit_max (Tensor): ``block_attn_res_prepare`` 输出的历史最大 logit，
            shape ``(T,)``，dtype 为 ``torch.float32``，且必须连续。
        exp_sum (Tensor): ``block_attn_res_prepare`` 输出的历史 softmax
            分母累积值，shape ``(T,)``，dtype 为 ``torch.float32``，且必须连续。
        eps (float64, optional): RMSNorm 数值稳定项，必须为有限正数，
            默认值为 ``1e-6``。以 double 传入 ACLNN，非空计算时转换为 float32 后仍须为有限正数。
        output_norm_weight (Tensor, optional): 下一子层 RMSNorm 的 gamma，shape ``(D,)``，
            dtype 为 BF16 或 FP32，必须连续且与其他输入在同一设备。``None`` 表示不启用输出 RMSNorm。
        output_norm_eps (float64, optional): 输出 RMSNorm 独立的 epsilon，默认 ``1e-6``；
            启用时转换为 FP32 前必须在 ``[FLT_MIN, FLT_MAX]`` 范围内，不支持次正规 epsilon。
            归一化保留原始 ``h`` 的 BF16 舍入边界，
            平方和、归一化及 gamma 乘法使用 FP32，最终输出 BF16。禁用时忽略本属性。

        当 ``T == 0`` 或 ``D == 0`` 时，所有输入仍必须满足上述 dimension 和
        shape 关系。ACLNN L0 直接返回 shape 与 ``delta`` 一致的空
        Tensor，不进入 InferShape、tiling 和 kernel。

    Returns:
        Tensor: 当前层结果 ``h``，shape 和 dtype 与 ``delta`` 一致，
        为连续 Tensor。输入 ``partial_block`` 在调用完成后已被原地更新。
    """
    return torch.ops.cann_ops_transformer.block_attn_res_update(
        partial_block,
        delta,
        pseudo_query,
        numerator,
        logit_max,
        exp_sum,
        eps=eps,
        output_norm_weight=output_norm_weight,
        output_norm_eps=output_norm_eps,
    )
