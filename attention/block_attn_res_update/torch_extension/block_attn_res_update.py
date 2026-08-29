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

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library

MATRIX_DIM_NUM = 2
VECTOR_DIM_NUM = 1
TOKEN_DIM_INDEX = 0
HIDDEN_DIM_INDEX = 1
MAX_HIDDEN_SIZE = 8192
DEFAULT_EPS = 1.0e-6


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
            f"float eps={DEFAULT_EPS}) -> Tensor"
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
) -> torch.Tensor:
    """block_attn_res_update(partial_block, delta, pseudo_query, numerator, logit_max, exp_sum, *, eps=1e-6) -> Tensor

    封装 ``block_attn_res_update`` 的 ``aclnnBlockAttnResUpdate`` 调用。

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
            默认值为 ``1e-6``。调用 ACLNN 前转换为 float32。

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
    )
