# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import get_as_library

get_as_library().define(
    "attn_epilogue("
    "Tensor x, Tensor woa, Tensor wob, Tensor descale_woa, Tensor descale_wob, "
    "Tensor rope_sin, Tensor rope_cos) -> Tensor"
)


@impl(get_as_library(), "attn_epilogue", "Meta")
def _attn_epilogue_meta(x, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos):
    del woa, wob, descale_woa, descale_wob, rope_sin, rope_cos
    return torch.empty((x.shape[0], 5120), dtype=torch.bfloat16, device="meta")


@impl(get_as_library(), "attn_epilogue", "PrivateUse1")
def attn_epilogue(
    x: torch.Tensor,
    woa: torch.Tensor,
    wob: torch.Tensor,
    descale_woa: torch.Tensor,
    descale_wob: torch.Tensor,
    rope_sin: torch.Tensor,
    rope_cos: torch.Tensor,
) -> torch.Tensor:
    from .op_dsl.attn_epilogue import attn_epilogue as _implementation

    return _implementation(x, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos)
