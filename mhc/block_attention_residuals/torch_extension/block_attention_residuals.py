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
from torch import _check as torch_check
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library

DEFAULT_NORM_EPS = 1.0e-6
PARTIAL_BLOCK_RANK = 2
BLOCK_RES_RANK = 3
NORM_WEIGHT_RANK = 1
PROJ_WEIGHT_RANK = 1
PROJ_WEIGHT_RANK_2D = 2
PROJ_WEIGHT_LEADING_ONES = 1
T_DIM_INDEX = 0
N_DIM_INDEX = 1
H_DIM_INDEX = 1
BLOCK_RES_H_DIM_INDEX = 2
MIN_HIDDEN_SIZE = 1
MIN_BLOCK_NUM = 1
MAX_BLOCK_NUM = 100
SUPPORTED_DTYPES = (torch.float16, torch.bfloat16, torch.float32)


def _check_inputs(
    partial_block,
    block_res,
    proj_weight,
    norm_weight,
    valid_block_num,
    norm_eps,
):
    torch_check(
        partial_block.dim() == PARTIAL_BLOCK_RANK,
        lambda: f"partial_block must be a 2D tensor, but got {partial_block.dim()}D",
    )
    torch_check(
        block_res.dim() == BLOCK_RES_RANK,
        lambda: f"block_res must be a 3D tensor, but got {block_res.dim()}D",
    )
    torch_check(
        norm_weight.dim() == NORM_WEIGHT_RANK,
        lambda: f"norm_weight must be a 1D tensor, but got {norm_weight.dim()}D",
    )
    torch_check(
        proj_weight.dim() == PROJ_WEIGHT_RANK
        or (
            proj_weight.dim() == PROJ_WEIGHT_RANK_2D
            and proj_weight.size(0) == PROJ_WEIGHT_LEADING_ONES
        ),
        lambda: f"proj_weight must be [H] or [1,H], but got {tuple(proj_weight.shape)}",
    )

    compute_dtype = partial_block.dtype
    torch_check(
        compute_dtype in SUPPORTED_DTYPES,
        lambda: (
            "partial_block dtype must be float16, bfloat16 or float32, "
            f"but got {compute_dtype}"
        ),
    )
    for tensor_name, tensor in (
        ("block_res", block_res),
        ("proj_weight", proj_weight),
        ("norm_weight", norm_weight),
    ):
        torch_check(
            tensor.dtype == compute_dtype,
            lambda tensor_name=tensor_name, tensor=tensor: (
                f"{tensor_name} dtype must match partial_block ({compute_dtype}), "
                f"but got {tensor.dtype}"
            ),
        )

    torch_check(
        partial_block.device
        == block_res.device
        == proj_weight.device
        == norm_weight.device,
        lambda: (
            "all inputs must be on the same device, but got "
            f"partial_block={partial_block.device}, block_res={block_res.device}, "
            f"proj_weight={proj_weight.device}, norm_weight={norm_weight.device}"
        ),
    )

    num_tokens = partial_block.size(T_DIM_INDEX)
    hidden_size = partial_block.size(H_DIM_INDEX)
    num_blocks = block_res.size(N_DIM_INDEX)
    torch_check(
        num_tokens >= 0
        and hidden_size >= MIN_HIDDEN_SIZE
        and MIN_BLOCK_NUM <= num_blocks <= MAX_BLOCK_NUM,
        lambda: (
            f"shape range requires T>=0, H>=1 and 1<=N<={MAX_BLOCK_NUM}, "
            f"but got T={num_tokens}, H={hidden_size}, N={num_blocks}"
        ),
    )
    torch_check(
        block_res.size(T_DIM_INDEX) == num_tokens
        and block_res.size(BLOCK_RES_H_DIM_INDEX) == hidden_size,
        lambda: f"block_res shape must be [T,N,H], but got {tuple(block_res.shape)}",
    )
    torch_check(
        proj_weight.size(-1) == hidden_size,
        lambda: (
            f"proj_weight last dim must equal H={hidden_size}, "
            f"but got {tuple(proj_weight.shape)}"
        ),
    )
    torch_check(
        norm_weight.size(0) == hidden_size,
        lambda: f"norm_weight must be [H], but got {tuple(norm_weight.shape)}",
    )
    torch_check(
        valid_block_num == -1 or valid_block_num == num_blocks,
        lambda: f"valid_block_num must be -1 or block_res.shape[1], got {valid_block_num}",
    )
    torch_check(
        math.isfinite(norm_eps) and norm_eps > 0.0,
        lambda: f"norm_eps must be finite and greater than zero, but got {norm_eps}",
    )


class BlockAttentionResidualsOpBuilder(OpBuilder):
    def __init__(self):
        super(BlockAttentionResidualsOpBuilder, self).__init__(
            "block_attention_residuals", category="mhc"
        )

    def sources(self):
        return ["csrc/mhc/block_attention_residuals.cpp"]

    def schema(self) -> str:
        return (
            "block_attention_residuals(Tensor partial_block, Tensor block_res, "
            "Tensor proj_weight, Tensor norm_weight, int valid_block_num=-1, "
            f"float norm_eps={DEFAULT_NORM_EPS}) -> Tensor"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def block_attention_residuals_meta(
            partial_block,
            block_res,
            proj_weight,
            norm_weight,
            valid_block_num=-1,
            norm_eps=DEFAULT_NORM_EPS,
        ):
            _check_inputs(
                partial_block,
                block_res,
                proj_weight,
                norm_weight,
                valid_block_num,
                norm_eps,
            )
            return torch.empty_like(partial_block, device="meta")


block_attention_residuals_op_builder = BlockAttentionResidualsOpBuilder()
block_attention_residuals_op_builder._ensure_initialized()


# Both the package-level torch.ops export and direct dispatcher calls use the
# same Python wrapper. The wrapper calls pybind directly, avoiding redispatch.
@impl(
    get_as_library(), block_attention_residuals_op_builder.name, "AutogradPrivateUse1"
)
@impl(get_as_library(), block_attention_residuals_op_builder.name, "PrivateUse1")
def _block_attention_residuals_dispatch(
    partial_block,
    block_res,
    proj_weight,
    norm_weight,
    valid_block_num=-1,
    norm_eps=DEFAULT_NORM_EPS,
):
    return block_attention_residuals(
        partial_block,
        block_res,
        proj_weight,
        norm_weight,
        valid_block_num,
        norm_eps,
    )


def block_attention_residuals(
    partial_block,
    block_res,
    proj_weight,
    norm_weight,
    valid_block_num=-1,
    norm_eps=DEFAULT_NORM_EPS,
):
    """Attention Residual 加权融合前向，封装 aclnnBlockAttentionResiduals。

    始终返回 hidden_states；需要梯度时保存中间结果并关联反向算子。
    """
    _check_inputs(
        partial_block, block_res, proj_weight, norm_weight, valid_block_num, norm_eps
    )
    needs_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad
        for tensor in (partial_block, block_res, proj_weight, norm_weight)
    )
    if needs_backward:
        return _BlockAttentionResidualsFunction.apply(
            partial_block,
            block_res,
            proj_weight,
            norm_weight,
            valid_block_num,
            norm_eps,
        )
    hidden, _, _ = _run_block_attention_residuals(
        partial_block,
        block_res,
        proj_weight,
        norm_weight,
        valid_block_num,
        norm_eps,
        False,
    )
    return hidden


def _run_block_attention_residuals(
    partial_block,
    block_res,
    proj_weight,
    norm_weight,
    valid_block_num,
    norm_eps,
    need_backward,
):
    op_module = block_attention_residuals_op_builder.load()
    hidden, inv_norm, probs = op_module.block_attention_residuals(
        partial_block,
        block_res,
        proj_weight,
        norm_weight,
        valid_block_num,
        norm_eps,
        need_backward,
    )
    return hidden, inv_norm, probs


class _BlockAttentionResidualsFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        partial_block,
        block_res,
        proj_weight,
        norm_weight,
        valid_block_num,
        norm_eps,
    ):
        hidden, inv_norm, probs = _run_block_attention_residuals(
            partial_block,
            block_res,
            proj_weight,
            norm_weight,
            valid_block_num,
            norm_eps,
            True,
        )
        ctx.save_for_backward(
            partial_block, block_res, proj_weight, norm_weight, inv_norm, probs
        )
        ctx.valid_block_num = valid_block_num
        return hidden

    @staticmethod
    def backward(ctx, grad_hidden_states):
        from cann_ops_transformer import block_attention_residuals_backward

        partial_block, block_res, proj_weight, norm_weight, inv_norm, probs = (
            ctx.saved_tensors
        )
        return (
            *block_attention_residuals_backward(
                partial_block,
                block_res,
                proj_weight,
                norm_weight,
                grad_hidden_states,
                inv_norm,
                probs,
                valid_block_num=ctx.valid_block_num,
            ),
            None,
            None,
        )
