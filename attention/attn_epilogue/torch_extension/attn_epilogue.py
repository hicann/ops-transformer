# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from collections import OrderedDict
import threading

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import get_as_library

from .op_dsl.attn_epilogue import (
    DIM,
    F,
    HEAD_DIM,
    MX_DIVISOR,
    O_LORA,
    _get_compiled,
    core_budget,
)

SCRATCH_LOCK = threading.RLock()
_SCRATCH = OrderedDict()
_CACHE_LIMIT = 8


def _projection_shape(x, woa, wob):
    """Read static dispatch metadata; callers supply tensors matching the documented contract."""
    t, heads, dim = x.shape
    if t < 1 or heads not in (8, 16, 64) or dim != HEAD_DIM:
        raise ValueError("x must have shape (T>=1, H, 512), H in (8, 16, 64)")
    return t, heads // 8, woa.shape[-1] == O_LORA, wob.shape[-1] == DIM


def _get_scratch(device, t, capture, ng):
    """Reuse four byte workspaces per device/stream/shape; capture allocates in its graph pool."""
    stream = torch.npu.current_stream(device)
    key = (device, stream.npu_stream, t, ng)
    # Capture gets graph-pool storage, independent of eager calls and other graphs.
    with SCRATCH_LOCK:
        if not capture and key in _SCRATCH:
            _SCRATCH.move_to_end(key)
            return _SCRATCH[key]
        m = (t + 15) // 16 * 16
        k = ng * O_LORA
        rows = ng * m
        specs = [
            ((rows, F), torch.uint8),
            ((rows, F // 32), torch.uint8),
            ((m, k), torch.uint8),
            ((m, k // 32), torch.uint8),
        ]
        buffers = tuple(
            torch.empty(shape, dtype=dtype, device=device) for shape, dtype in specs
        )
        if not capture:
            _SCRATCH[key] = buffers
            if len(_SCRATCH) > _CACHE_LIMIT:
                _SCRATCH.popitem(last=False)
        return buffers


get_as_library().define(
    "attn_epilogue("
    "Tensor x, Tensor woa, Tensor wob, Tensor descale_woa, Tensor descale_wob, "
    "Tensor rope_sin, Tensor rope_cos) -> Tensor"
)


@impl(get_as_library(), "attn_epilogue", "Meta")
def _attn_epilogue_meta(x, woa, wob, descale_woa, descale_wob, rope_sin, rope_cos):
    del woa, wob, descale_woa, descale_wob, rope_sin, rope_cos
    return torch.empty((x.shape[0], DIM), dtype=torch.bfloat16, device="meta")


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
    """Project BF16 [T, NG*8, 512] to [T, 5120] using prequantized ND/NZ weights.

    The caller supplies contiguous tensors on one NPU with the shapes/dtypes
    documented in README. Only activation dimensions are checked here; tensor
    metadata and private-format scans are intentionally absent from the hot path.
    """
    t, ng, first_nz, second_nz = _projection_shape(x, woa, wob)
    m = (t + 15) // 16 * 16
    with torch.npu.device(x.device):
        scratch = _get_scratch(x.device, t, torch.npu.is_current_stream_capturing(), ng)
        out = torch.empty((m, DIM), dtype=torch.bfloat16, device=x.device)
        _get_compiled(t, ng, first_nz, second_nz, m, core_budget())(
            x.view(t * ng, F),
            rope_cos,
            rope_sin,
            woa if first_nz else woa.view(ng * O_LORA, F),
            descale_woa.view(torch.uint8)
            if first_nz
            else descale_woa.view(torch.uint8).view(ng * O_LORA, F // MX_DIVISOR, 2),
            wob,
            descale_wob.view(torch.uint8),
            scratch[0],
            scratch[1],
            scratch[2],
            scratch[3],
            out,
        )
        return out[:t]
