# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
import os
from typing import Optional, Tuple

import torch
from torch.library import impl
from torch_npu.utils._error_code import ErrCode, ops_error
from cann_ops_transformer.op_builder import OpBuilder, get_as_library


_QUANT_MODE_NONE = -1
_QUANT_MODE_MXFP8_E5M2 = 2
_QUANT_MODE_MXFP8_E4M3FN = 3
_QUANT_MODE_MXFP4_E2M1 = 9
_MX_QUANT_BLOCK_SIZE = 32
_MXFP4_SCALE_BLOCK_SIZE = 64
_PAD_TO_EVEN_FACTOR = 2


def _is_mx_quant_mode(quant_mode: int) -> bool:
    return quant_mode in (
        _QUANT_MODE_MXFP8_E5M2,
        _QUANT_MODE_MXFP8_E4M3FN,
        _QUANT_MODE_MXFP4_E2M1,
    )


def _actual_num_out_tokens(indices: torch.Tensor, num_out_tokens: Optional[int]) -> int:
    flatten_size = indices.numel()
    num_out_tokens_value = 0 if num_out_tokens is None else num_out_tokens
    if num_out_tokens_value > 0:
        return min(num_out_tokens_value, flatten_size)
    return max(num_out_tokens_value + flatten_size, 0)


def _align_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


class MoeTokenPermuteOpBuilder(OpBuilder):
    def __init__(self):
        super(MoeTokenPermuteOpBuilder, self).__init__(
            "moe_token_permute", category="moe"
        )

    def sources(self):
        return ["csrc/moe/moe_token_permute.cpp"]

    def include_paths(self):
        paths = super().include_paths()
        paths.append(
            os.path.abspath(
                os.path.join(
                    self._package_path,
                    "..",
                    "..",
                    "moe",
                    "moe_token_permute",
                    "op_host",
                    "op_api",
                )
            )
        )
        return paths

    def schema(self) -> str:
        return (
            "moe_token_permute(Tensor tokens, Tensor indices, "
            "int? num_out_tokens=None, bool padded_mode=False, int quant_mode=-1) -> (Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def moe_token_permute_meta(
            tokens, indices, num_out_tokens=None, padded_mode=False, quant_mode=-1
        ):
            torch._check(
                tokens.dim() == 2,
                lambda: f"tokens must be a 2D tensor, but got {tokens.dim()} dimensions."
                f"{ops_error(ErrCode.VALUE)}",
            )
            torch._check(
                indices.dim() in (1, 2),
                lambda: f"indices must be a 1D or 2D tensor, but got {indices.dim()} dimensions."
                f"{ops_error(ErrCode.VALUE)}",
            )
            torch._check(
                quant_mode == _QUANT_MODE_NONE or _is_mx_quant_mode(quant_mode),
                lambda: "quant_mode only supports -1, 2, 3 or 9."
                f"{ops_error(ErrCode.VALUE)}",
            )

            out_tokens = _actual_num_out_tokens(indices, num_out_tokens)
            hidden_size = tokens.size(1)
            sorted_indices = indices.new_empty((indices.numel(),), dtype=torch.int32)

            if quant_mode == _QUANT_MODE_MXFP8_E5M2:
                permuted_tokens = tokens.new_empty(
                    (out_tokens, hidden_size), dtype=torch.float8_e5m2
                )
                scale_cols = _align_up(
                    _ceil_div(hidden_size, _MX_QUANT_BLOCK_SIZE), _PAD_TO_EVEN_FACTOR
                )
                expanded_scale = tokens.new_empty(
                    (out_tokens, scale_cols), dtype=torch.float8_e8m0fnu
                )
            elif quant_mode == _QUANT_MODE_MXFP8_E4M3FN:
                permuted_tokens = tokens.new_empty(
                    (out_tokens, hidden_size), dtype=torch.float8_e4m3fn
                )
                scale_cols = _align_up(
                    _ceil_div(hidden_size, _MX_QUANT_BLOCK_SIZE), _PAD_TO_EVEN_FACTOR
                )
                expanded_scale = tokens.new_empty(
                    (out_tokens, scale_cols), dtype=torch.float8_e8m0fnu
                )
            elif quant_mode == _QUANT_MODE_MXFP4_E2M1:
                torch._check(
                    hidden_size % 2 == 0,
                    lambda: "The hidden size must be even when quant_mode is 9."
                    f"{ops_error(ErrCode.VALUE)}",
                )
                permuted_tokens = tokens.new_empty(
                    (out_tokens, hidden_size // 2), dtype=torch.uint8
                )
                expanded_scale = tokens.new_empty(
                    (out_tokens, _ceil_div(hidden_size, _MXFP4_SCALE_BLOCK_SIZE), 2),
                    dtype=torch.uint8,
                )
            else:
                permuted_tokens = tokens.new_empty(
                    (out_tokens, hidden_size), dtype=tokens.dtype
                )
                expanded_scale = tokens.new_empty((0,), dtype=torch.float32)

            return permuted_tokens, sorted_indices, expanded_scale


moe_token_permute_op_builder = MoeTokenPermuteOpBuilder()
moe_token_permute_op_builder._ensure_initialized()


@impl(get_as_library(), moe_token_permute_op_builder.name, "PrivateUse1")
def _moe_token_permute(
    tokens, indices, num_out_tokens=None, padded_mode=False, quant_mode=-1
):
    op_module = moe_token_permute_op_builder.load()
    return op_module.moe_token_permute(
        tokens, indices, num_out_tokens, padded_mode, quant_mode
    )


def moe_token_permute(
    tokens: torch.Tensor,
    indices: torch.Tensor,
    num_out_tokens: Optional[int] = None,
    padded_mode: bool = False,
    quant_mode: int = _QUANT_MODE_NONE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.ops.cann_ops_transformer.moe_token_permute(
        tokens,
        indices,
        num_out_tokens,
        padded_mode,
        quant_mode,
    )
