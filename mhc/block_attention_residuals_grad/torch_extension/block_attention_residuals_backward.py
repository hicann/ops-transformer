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
from torch import _check as torch_check
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library

MAIN_DTYPES = (torch.float16, torch.bfloat16, torch.float32)
DEFAULT_VALID_BLOCK_NUM = 0
DIM_INDEX_BATCH = 0
DIM_INDEX_BLOCK = 1
DIM_INDEX_HIDDEN = 1
DIM_INDEX_BLOCK_RES_HIDDEN = 2
MIN_TOKEN_NUM = 1
MIN_BLOCK_NUM = 0
MAX_BLOCK_NUM = 128
MIN_HIDDEN_SIZE = 1


def _check_dtypes(
    partial_block: torch.Tensor,
    block_res: torch.Tensor,
    proj_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    grad_hidden_states: torch.Tensor,
    inv_norm: torch.Tensor,
    probs: torch.Tensor,
) -> None:
    main_tensors = (
        ("partial_block", partial_block),
        ("block_res", block_res),
        ("proj_weight", proj_weight),
        ("norm_weight", norm_weight),
        ("grad_hidden_states", grad_hidden_states),
    )
    for tensor_name, tensor in main_tensors:
        torch_check(
            tensor.dtype in MAIN_DTYPES,
            lambda tensor_name=tensor_name, tensor=tensor: (
                f"{tensor_name} dtype must be float16/bfloat16/float32, but got {tensor.dtype}"
            ),
        )
    for tensor_name, tensor in main_tensors[1:]:
        torch_check(
            tensor.dtype == partial_block.dtype,
            lambda tensor_name=tensor_name, tensor=tensor: (
                f"{tensor_name} dtype must be same as partial_block dtype, "
                f"but got {tensor.dtype} and {partial_block.dtype}"
            ),
        )
    for tensor_name, tensor in (("inv_norm", inv_norm), ("probs", probs)):
        torch_check(
            tensor.dtype == torch.float32,
            lambda tensor_name=tensor_name, tensor=tensor: (
                f"{tensor_name} dtype must be float32, but got {tensor.dtype}"
            ),
        )


def _check_dimensions(
    partial_block: torch.Tensor,
    block_res: torch.Tensor,
    proj_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    grad_hidden_states: torch.Tensor,
    inv_norm: torch.Tensor,
    probs: torch.Tensor,
) -> None:
    tensor_dimensions = (
        ("partial_block", partial_block, 2),
        ("block_res", block_res, 3),
        ("proj_weight", proj_weight, 2),
        ("norm_weight", norm_weight, 1),
        ("grad_hidden_states", grad_hidden_states, 2),
        ("inv_norm", inv_norm, 2),
        ("probs", probs, 2),
    )
    for tensor_name, tensor, expected_rank in tensor_dimensions:
        torch_check(
            tensor.dim() == expected_rank,
            lambda tensor_name=tensor_name,
            tensor=tensor,
            expected_rank=expected_rank: (
                f"{tensor_name} must be a {expected_rank}D tensor, but got {tensor.dim()}D"
            ),
        )


def _check_shapes(
    partial_block: torch.Tensor,
    block_res: torch.Tensor,
    proj_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    grad_hidden_states: torch.Tensor,
    inv_norm: torch.Tensor,
    probs: torch.Tensor,
) -> None:
    token_num = partial_block.size(DIM_INDEX_BATCH)
    hidden_size = partial_block.size(DIM_INDEX_HIDDEN)
    block_num = block_res.size(DIM_INDEX_BLOCK)
    torch_check(
        token_num >= MIN_TOKEN_NUM,
        lambda: f"partial_block.size(0) must be >= {MIN_TOKEN_NUM}, but got {token_num}",
    )
    torch_check(
        hidden_size >= MIN_HIDDEN_SIZE,
        lambda: f"partial_block.size(1) must be >= {MIN_HIDDEN_SIZE}, but got {hidden_size}",
    )
    torch_check(
        MIN_BLOCK_NUM <= block_num <= MAX_BLOCK_NUM,
        lambda: f"block_res.size(1) must be in [{MIN_BLOCK_NUM}, {MAX_BLOCK_NUM}], but got {block_num}",
    )
    torch_check(
        block_res.size(DIM_INDEX_BATCH) == token_num,
        lambda: (
            "block_res.size(0) must equal partial_block.size(0), but got "
            f"{block_res.size(DIM_INDEX_BATCH)} and {token_num}"
        ),
    )
    torch_check(
        block_res.size(DIM_INDEX_BLOCK_RES_HIDDEN) == hidden_size,
        lambda: (
            "block_res.size(2) must equal partial_block.size(1), but got "
            f"{block_res.size(DIM_INDEX_BLOCK_RES_HIDDEN)} and {hidden_size}"
        ),
    )
    torch_check(
        proj_weight.size(DIM_INDEX_BATCH) == 1
        and proj_weight.size(DIM_INDEX_HIDDEN) == hidden_size,
        lambda: f"proj_weight must have shape [1, {hidden_size}], but got {proj_weight.shape}",
    )
    torch_check(
        norm_weight.size(0) == hidden_size,
        lambda: f"norm_weight must have shape [{hidden_size}], but got {norm_weight.shape}",
    )
    torch_check(
        grad_hidden_states.size(DIM_INDEX_BATCH) == token_num
        and grad_hidden_states.size(DIM_INDEX_HIDDEN) == hidden_size,
        lambda: f"grad_hidden_states must have shape [{token_num}, {hidden_size}], but got {grad_hidden_states.shape}",
    )
    for tensor_name, tensor in (("inv_norm", inv_norm), ("probs", probs)):
        torch_check(
            tensor.size(DIM_INDEX_BATCH) == token_num
            and tensor.size(DIM_INDEX_BLOCK) == block_num + 1,
            lambda tensor_name=tensor_name, tensor=tensor: (
                f"{tensor_name} must have shape [{token_num}, {block_num + 1}], but got {tensor.shape}"
            ),
        )


def _check_device(
    partial_block: torch.Tensor,
    block_res: torch.Tensor,
    proj_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    grad_hidden_states: torch.Tensor,
    inv_norm: torch.Tensor,
    probs: torch.Tensor,
) -> None:
    tensor_names = (
        ("partial_block", partial_block),
        ("block_res", block_res),
        ("proj_weight", proj_weight),
        ("norm_weight", norm_weight),
        ("grad_hidden_states", grad_hidden_states),
        ("inv_norm", inv_norm),
        ("probs", probs),
    )
    for tensor_name, tensor in tensor_names[1:]:
        torch_check(
            tensor.device == partial_block.device,
            lambda tensor_name=tensor_name, tensor=tensor: (
                "all inputs must be on the same device, but got "
                f"{tensor_name}={tensor.device} and partial_block={partial_block.device}"
            ),
        )


def _check_inputs(
    partial_block: torch.Tensor,
    block_res: torch.Tensor,
    proj_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    grad_hidden_states: torch.Tensor,
    inv_norm: torch.Tensor,
    probs: torch.Tensor,
) -> None:
    _check_dimensions(
        partial_block,
        block_res,
        proj_weight,
        norm_weight,
        grad_hidden_states,
        inv_norm,
        probs,
    )
    _check_dtypes(
        partial_block,
        block_res,
        proj_weight,
        norm_weight,
        grad_hidden_states,
        inv_norm,
        probs,
    )
    _check_shapes(
        partial_block,
        block_res,
        proj_weight,
        norm_weight,
        grad_hidden_states,
        inv_norm,
        probs,
    )
    _check_device(
        partial_block,
        block_res,
        proj_weight,
        norm_weight,
        grad_hidden_states,
        inv_norm,
        probs,
    )


class BlockAttentionResidualsBackwardOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("block_attention_residuals_backward", category="mhc")

    def sources(self):
        return ["csrc/mhc/block_attention_residuals_backward.cpp"]

    def schema(self) -> str:
        return (
            "block_attention_residuals_backward(Tensor partial_block, Tensor block_res, "
            "Tensor proj_weight, Tensor norm_weight, Tensor grad_hidden_states, "
            "Tensor inv_norm, Tensor probs, *, int valid_block_num=0) -> "
            "(Tensor, Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def block_attention_residuals_backward_meta(
            partial_block,
            block_res,
            proj_weight,
            norm_weight,
            grad_hidden_states,
            inv_norm,
            probs,
            *,
            valid_block_num=DEFAULT_VALID_BLOCK_NUM,
        ):
            _check_inputs(
                partial_block,
                block_res,
                proj_weight,
                norm_weight,
                grad_hidden_states,
                inv_norm,
                probs,
            )
            meta_options = {"dtype": partial_block.dtype, "device": "meta"}
            grad_partial_block = torch.empty(partial_block.shape, **meta_options)
            grad_block_res = torch.empty(block_res.shape, **meta_options)
            grad_proj_weight = torch.empty(proj_weight.shape, **meta_options)
            grad_norm_weight = torch.empty(norm_weight.shape, **meta_options)
            return (
                grad_partial_block,
                grad_block_res,
                grad_proj_weight,
                grad_norm_weight,
            )


block_attention_residuals_backward_op_builder = (
    BlockAttentionResidualsBackwardOpBuilder()
)
block_attention_residuals_backward_op_builder._ensure_initialized()


@impl(
    get_as_library(), block_attention_residuals_backward_op_builder.name, "PrivateUse1"
)
def _block_attention_residuals_backward_dispatch(
    partial_block: torch.Tensor,
    block_res: torch.Tensor,
    proj_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    grad_hidden_states: torch.Tensor,
    inv_norm: torch.Tensor,
    probs: torch.Tensor,
    *,
    valid_block_num: int = DEFAULT_VALID_BLOCK_NUM,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    op_module = block_attention_residuals_backward_op_builder.load()
    return op_module.block_attention_residuals_backward(
        partial_block,
        block_res,
        proj_weight,
        norm_weight,
        grad_hidden_states,
        inv_norm,
        probs,
        valid_block_num,
    )


def block_attention_residuals_backward(
    partial_block: torch.Tensor,
    block_res: torch.Tensor,
    proj_weight: torch.Tensor,
    norm_weight: torch.Tensor,
    grad_hidden_states: torch.Tensor,
    inv_norm: torch.Tensor,
    probs: torch.Tensor,
    *,
    valid_block_num: int = DEFAULT_VALID_BLOCK_NUM,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run BlockAttentionResidualsGrad on NPU.

    The operator fuses softmax backward, RMS-normalization backward and the
    attention weighted-sum backward, producing the four gradients saved by the
    BlockAttentionResiduals forward operator.

    Args:
        partial_block (Tensor): FP16/BF16/FP32 tensor of shape ``[T, H]``.
        block_res (Tensor): FP16/BF16/FP32 tensor of shape ``[T, N, H]``.
        proj_weight (Tensor): FP16/BF16/FP32 tensor of shape ``[1, H]``.
        norm_weight (Tensor): FP16/BF16/FP32 tensor of shape ``[H]``.
        grad_hidden_states (Tensor): FP16/BF16/FP32 tensor of shape ``[T, H]``.
        inv_norm (Tensor): FP32 tensor of shape ``[T, N + 1]``.
        probs (Tensor): FP32 tensor of shape ``[T, N + 1]``.
        valid_block_num (int): Reserved attribute; defaults to 0 and is not used
            by the current kernel.

    Returns:
        Tuple[Tensor, Tensor, Tensor, Tensor]: gradients of partial_block,
        block_res, proj_weight and norm_weight with shapes identical to the
        corresponding inputs.
    """
    _check_inputs(
        partial_block,
        block_res,
        proj_weight,
        norm_weight,
        grad_hidden_states,
        inv_norm,
        probs,
    )
    return torch.ops.cann_ops_transformer.block_attention_residuals_backward(
        partial_block,
        block_res,
        proj_weight,
        norm_weight,
        grad_hidden_states,
        inv_norm,
        probs,
        valid_block_num=valid_block_num,
    )
