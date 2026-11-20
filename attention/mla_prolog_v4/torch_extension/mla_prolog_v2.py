# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from enum import Enum
from typing import Optional, Tuple

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import get_as_library


class CacheMode(Enum):
    PA_BSND = 0
    PA_NZ = 1


class QkvQuantMode(Enum):
    A16C16 = 0
    A16C8_KV_NOPE_FP8_E4M3_PER_TOKEN_GROUP_GROUP_SIZE_128_ROPE_BF16_SCALE_FLOAT_KV_CACHE_LAYOUT_N_NOPE_ROPE_SCALE = 1
    A8C8_Q_FP8_E4M3_PER_TOKEN_HEAD_KV_FP8_E4M3_PER_TENSOR = 2
    A8C8_QKV_HIFLOAT8_PER_TENSOR = 3


class RopeMode(Enum):
    NO_ROPE = 0
    ROPE_INTERLEAVE_HALF = 1
    ROPE_HALF = 2


_QKV_QUANT_MODE_VALUES = tuple((_m.value for _m in QkvQuantMode))
_CACHE_MODE_VALUES = tuple((_m.value for _m in CacheMode))
_ROPE_MODE_VALUES = tuple((_m.value for _m in RopeMode))


def _validate_mode_enums(qkv_quant_mode, cache_mode, rope_mode):
    for name, value, enum_type in (
        ("qkv_quant_mode", qkv_quant_mode, QkvQuantMode),
        ("cache_mode", cache_mode, CacheMode),
        ("rope_mode", rope_mode, RopeMode),
    ):
        if type(value) is not enum_type:
            raise TypeError(f"{name} must be a {enum_type.__name__} enum")


def _validate_scalar_types(norm_eps, qr_scale, kvr_scale):
    for name, value in (
        ("norm_eps", norm_eps),
        ("qr_scale", qr_scale),
        ("kvr_scale", kvr_scale),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a finite float")


_TORCH_SCHEMA = "mla_prolog_v2(Tensor x, Tensor wqa, Tensor wqb, Tensor wkva, Tensor wkb, Tensor norm_weight_qa, Tensor norm_weight_kva, Tensor(a!) kv_cache, Tensor cache_index, int qkv_quant_mode, *, Tensor? quant_scale_kv=None, Tensor? rope_sin=None, Tensor? rope_cos=None, float norm_eps=1e-5, int cache_mode=0, float qr_scale=1.0, float kvr_scale=1.0, int rope_mode=1, bool return_qr=False) -> (Tensor, Tensor, Tensor)"


def _validate_schema_modes(qkv_quant_mode, cache_mode, rope_mode):
    for name, value, allowed in (
        ("qkv_quant_mode", qkv_quant_mode, _QKV_QUANT_MODE_VALUES),
        ("cache_mode", cache_mode, _CACHE_MODE_VALUES),
        ("rope_mode", rope_mode, _ROPE_MODE_VALUES),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} schema value must be int")
        if value not in allowed:
            raise ValueError(f"{name} has unsupported schema value {value}")


def mla_prolog_v2_fake(
    x: torch.Tensor,
    wqa: torch.Tensor,
    wqb: torch.Tensor,
    wkva: torch.Tensor,
    wkb: torch.Tensor,
    norm_weight_qa: torch.Tensor,
    norm_weight_kva: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_index: torch.Tensor,
    qkv_quant_mode: int,
    *,
    quant_scale_kv: Optional[torch.Tensor] = None,
    rope_sin: Optional[torch.Tensor] = None,
    rope_cos: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-05,
    cache_mode: int = 0,
    qr_scale: float = 1.0,
    kvr_scale: float = 1.0,
    rope_mode: int = 1,
    return_qr: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Describe fixed output slots without reading device data or launching DSL."""
    _validate_schema_modes(qkv_quant_mode, cache_mode, rope_mode)
    (t, heads, dr) = (x.shape[0], wkb.shape[0], 0 if wkb.shape[1] == 256 else 64)
    q_dtype = (
        torch.bfloat16
        if qkv_quant_mode < 2
        else torch.float8_e4m3fn
        if qkv_quant_mode == 2
        else torch.uint8
    )
    scale_shape = (
        (0,)
        if qkv_quant_mode < 2 or (t == 0 and qkv_quant_mode == 3)
        else (t, heads, 1)
        if qkv_quant_mode == 2
        else (1,)
    )
    return (
        x.new_empty((t, heads, 512 + dr), dtype=q_dtype),
        x.new_empty(scale_shape, dtype=torch.float32),
        x.new_empty(
            (t, norm_weight_qa.shape[0]) if return_qr else (0,), dtype=torch.bfloat16
        ),
    )


def mla_prolog_v2(
    x: torch.Tensor,
    wqa: torch.Tensor,
    wqb: torch.Tensor,
    wkva: torch.Tensor,
    wkb: torch.Tensor,
    norm_weight_qa: torch.Tensor,
    norm_weight_kva: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_index: torch.Tensor,
    qkv_quant_mode: QkvQuantMode,
    *,
    quant_scale_kv: Optional[torch.Tensor] = None,
    rope_sin: Optional[torch.Tensor] = None,
    rope_cos: Optional[torch.Tensor] = None,
    norm_eps: float = 1e-05,
    cache_mode: CacheMode = CacheMode.PA_BSND,
    qr_scale: float = 1.0,
    kvr_scale: float = 1.0,
    rope_mode: RopeMode = RopeMode.ROPE_INTERLEAVE_HALF,
    return_qr: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project BF16 MLA inputs, apply RMSNorm/RoPE and update ``kv_cache``.

    ``x`` has shape ``(T, dim)``; weights use the selected ND/NZ layout.
    ``norm_weight_qa`` and ``norm_weight_kva`` are FP32
    RMSNorm weights; ``cache_index`` selects the slots updated in ``kv_cache``.
    Optional quantization scale and RoPE tensors are keyword arguments.
    Mode attributes use the enums attached to this function. ``norm_eps``,
    ``qr_scale`` and ``kvr_scale`` control normalization and output scaling.

    Returns ``(q, descale_q, qr)``. ``return_qr`` controls whether ``qr`` is empty.
    Cache is updated in place and is not included in the return values.
    """
    _validate_mode_enums(qkv_quant_mode, cache_mode, rope_mode)
    if type(return_qr) is not bool:
        raise TypeError("return_qr must be bool")
    _validate_scalar_types(norm_eps, qr_scale, kvr_scale)
    dispatch = (
        torch.ops.cann_ops_transformer._mla_prolog_v2_graph.default
        if torch.compiler.is_compiling()
        or (
            isinstance(x, torch.Tensor)
            and x.device.type in ("npu", "privateuseone")
            and torch.npu.is_current_stream_capturing()
        )
        else torch.ops.cann_ops_transformer.mla_prolog_v2.default
    )
    return dispatch(
        x,
        wqa,
        wqb,
        wkva,
        wkb,
        norm_weight_qa,
        norm_weight_kva,
        kv_cache,
        cache_index,
        quant_scale_kv=quant_scale_kv,
        rope_sin=rope_sin,
        rope_cos=rope_cos,
        norm_eps=norm_eps,
        qkv_quant_mode=qkv_quant_mode.value,
        cache_mode=cache_mode.value,
        qr_scale=qr_scale,
        kvr_scale=kvr_scale,
        rope_mode=rope_mode.value,
        return_qr=return_qr,
    )


get_as_library().define(_TORCH_SCHEMA)
get_as_library().define(
    _TORCH_SCHEMA.replace("mla_prolog_v2(", "_mla_prolog_v2_graph(", 1)
)
impl(get_as_library(), "mla_prolog_v2", "Meta")(mla_prolog_v2_fake)
impl(get_as_library(), "_mla_prolog_v2_graph", "Meta")(mla_prolog_v2_fake)


@impl(get_as_library(), "mla_prolog_v2", "PrivateUse1")
def _dispatch_eager(*values, **attributes):
    from .op_dsl import mla_prolog_v4 as implementation

    return implementation.mla_prolog_v2_npu(*values, **attributes)


@impl(get_as_library(), "_mla_prolog_v2_graph", "PrivateUse1")
def _dispatch_graph(*values, **attributes):
    from .op_dsl import mla_prolog_v4 as implementation

    return implementation.mla_prolog_v2_graph_npu(*values, **attributes)


mla_prolog_v2.QkvQuantMode = QkvQuantMode
mla_prolog_v2.CacheMode = CacheMode
mla_prolog_v2.RopeMode = RopeMode
