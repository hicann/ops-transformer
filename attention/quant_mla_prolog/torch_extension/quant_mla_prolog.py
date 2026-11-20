# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from enum import Enum as _Enum
import math as _math
import struct as _struct

import torch as _torch
from torch.library import impl

from cann_ops_transformer.op_builder import get_as_library


KV_ELEMENTS = 512


class WQrQuantMode(_Enum):
    A8W8_MXFP8 = 1
    A8W8_HIFLOAT8_PER_TENSOR = 2
    A4W4_HIFLOAT4 = 3


class QkvQuantMode(_Enum):
    A16C16 = 0
    A16C8_KV_NOPE_FP8_E4M3_PER_TOKEN_GROUP_GROUP_SIZE_128_ROPE_BF16_SCALE_FLOAT_KV_CACHE_LAYOUT_N_NOPE_ROPE_SCALE = 1
    A8C8_Q_FP8_E4M3_PER_TOKEN_HEAD_KV_FP8_E4M3_PER_TENSOR = 2
    A8C8_QKV_HIFLOAT8_PER_TENSOR = 3


class CacheMode(_Enum):
    PA_BSND = 0
    PA_NZ = 1


class RopeMode(_Enum):
    NO_ROPE = 0
    ROPE_INTERLEAVE_HALF = 1
    ROPE_HALF = 2


_W_QR_QUANT_MODE_VALUES = tuple((_m.value for _m in WQrQuantMode))
_QKV_QUANT_MODE_VALUES = tuple((_m.value for _m in QkvQuantMode))
_CACHE_MODE_VALUES = tuple((_m.value for _m in CacheMode))
_ROPE_MODE_VALUES = tuple((_m.value for _m in RopeMode))
_VALID_PAIRS = {(0, 1), (0, 2), (1, 1), (2, 1), (3, 2)}
_E4 = _torch.float8_e4m3fn
_E8 = _torch.float8_e8m0fnu


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _mode_value(name, value, enum_type):
    if type(value) is not enum_type:
        raise TypeError(f"{name} must be a {enum_type.__name__} enum")
    return value.value


def _validate_modes(w_qr_quant_mode, qkv_quant_mode, cache_mode, rope_mode):
    for name, value, allowed in (
        ("w_qr_quant_mode", w_qr_quant_mode, _W_QR_QUANT_MODE_VALUES),
        ("qkv_quant_mode", qkv_quant_mode, _QKV_QUANT_MODE_VALUES),
        ("cache_mode", cache_mode, _CACHE_MODE_VALUES),
        ("rope_mode", rope_mode, _ROPE_MODE_VALUES),
    ):
        if type(value) is not int:
            raise TypeError(f"{name} schema value must be int")
        if value not in allowed:
            raise ValueError(f"{name} has unsupported schema value {value}")
    if w_qr_quant_mode == WQrQuantMode.A4W4_HIFLOAT4.value:
        raise NotImplementedError("A4W4_HIFLOAT4 is not implemented")
    _require(
        (qkv_quant_mode, w_qr_quant_mode) in _VALID_PAIRS,
        "unsupported qkv_quant_mode/w_qr_quant_mode combination",
    )
    _require(
        not (
            cache_mode == CacheMode.PA_NZ.value
            and qkv_quant_mode
            == QkvQuantMode.A16C8_KV_NOPE_FP8_E4M3_PER_TOKEN_GROUP_GROUP_SIZE_128_ROPE_BF16_SCALE_FLOAT_KV_CACHE_LAYOUT_N_NOPE_ROPE_SCALE.value
        ),
        "qkv_quant_mode=1 does not support PA_NZ",
    )


def _output_specs(T, N, q_lora, Dr, w_qr_quant_mode, qkv_quant_mode, return_qr):
    q_dtype = (
        _torch.bfloat16
        if qkv_quant_mode in (0, 1)
        else _E4
        if qkv_quant_mode == 2
        else _torch.uint8
    )
    qr_dtype = _E4 if w_qr_quant_mode == 1 else _torch.uint8
    descale_qr_dtype = _E8 if w_qr_quant_mode == 1 else _torch.float32
    descale_q_shape = (
        (T, N, 1)
        if qkv_quant_mode == 2
        else (int(T != 0),)
        if qkv_quant_mode == 3
        else (0,)
    )
    qr_shape = (T, q_lora) if return_qr else (0,)
    descale_qr_shape = (
        ((T, q_lora // 32) if w_qr_quant_mode == 1 else (int(T != 0),))
        if return_qr
        else (0,)
    )
    return (
        ((T, N, KV_ELEMENTS + Dr), q_dtype),
        (descale_q_shape, _torch.float32),
        (qr_shape, qr_dtype),
        (descale_qr_shape, descale_qr_dtype),
    )


def _validate_scalars(norm_eps, qr_scale, kvr_scale):
    for name, value in (
        ("norm_eps", norm_eps),
        ("qr_scale", qr_scale),
        ("kvr_scale", kvr_scale),
    ):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a Python float or int, not bool or Tensor")
        try:
            fp32_value = _struct.unpack("f", _struct.pack("f", float(value)))[0]
        except (OverflowError, _struct.error) as error:
            raise ValueError(f"{name} must be finite in FP32") from error
        _require(_math.isfinite(fp32_value), f"{name} must be finite in FP32")
        if name == "norm_eps":
            _require(fp32_value > 0, f"{name} must be strictly positive in FP32")


def _validate_rope_inputs(rope_sin, rope_cos, T, Dr, rope_mode, device, *, fake=False):
    if rope_sin is None and rope_cos is None:
        _require(
            not (Dr and rope_mode), "rope_sin/rope_cos are required for active RoPE"
        )
        return
    _require(
        rope_sin is not None and rope_cos is not None,
        "rope_sin/rope_cos must be supplied together",
    )
    for name, value in (("rope_sin", rope_sin), ("rope_cos", rope_cos)):
        _require(isinstance(value, _torch.Tensor), f"{name} must be a Tensor")
        _require(tuple(value.shape) == (T, Dr), f"{name} shape must be {(T, Dr)}")
        _require(value.dtype == _torch.bfloat16, f"{name} must be BF16")
        _require(value.device == device, f"{name} must be on {device}")
        _require(value.is_contiguous(), f"{name} must be contiguous")


_TORCH_SCHEMA = "quant_mla_prolog(Tensor x, Tensor wqa, Tensor wqb, Tensor wkva, Tensor wkb, Tensor descale_x, Tensor descale_wqa, Tensor descale_wqb, Tensor descale_wkva, Tensor norm_weight_qa, Tensor norm_weight_kva, Tensor(a!) kv_cache, Tensor cache_index, int w_qr_quant_mode, int qkv_quant_mode, *, Tensor? quant_scale_kv=None, Tensor? rope_sin=None, Tensor? rope_cos=None, float norm_eps=1e-5, int cache_mode=0, float qr_scale=1., float kvr_scale=1., int rope_mode=1, bool return_qr=False) -> (Tensor, Tensor, Tensor, Tensor)"


def _fake(
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    descale_x,
    descale_wqa,
    descale_wqb,
    descale_wkva,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    w_qr_quant_mode,
    qkv_quant_mode,
    *,
    quant_scale_kv=None,
    rope_sin=None,
    rope_cos=None,
    norm_eps=1e-05,
    cache_mode=0,
    qr_scale=1.0,
    kvr_scale=1.0,
    rope_mode=1,
    return_qr=False,
):
    _validate_modes(w_qr_quant_mode, qkv_quant_mode, cache_mode, rope_mode)
    _require(type(return_qr) is bool, "return_qr must be bool")
    _validate_scalars(norm_eps, qr_scale, kvr_scale)
    T = x.shape[0]
    N = wkb.shape[0]
    Dr = 0 if wkb.shape[1] == 256 else 64
    q_lora = norm_weight_qa.shape[0]
    _require(
        Dr != 0 or rope_mode == RopeMode.NO_ROPE.value, "Dr=0 requires rope_mode=0"
    )
    _validate_rope_inputs(rope_sin, rope_cos, T, Dr, rope_mode, x.device, fake=True)
    return tuple(
        (
            x.new_empty(shape, dtype=dtype)
            for (shape, dtype) in _output_specs(
                T, N, q_lora, Dr, w_qr_quant_mode, qkv_quant_mode, return_qr
            )
        )
    )


def _reject_functionalization(*args, **kwargs):
    raise RuntimeError(
        "quant_mla_prolog requires a backend preserving the mutable cache op; functionalization with cache clone/copy-back is disabled"
    )


def _graph_fake(*args, **attrs):
    (q, descale_q, qr, descale_qr) = _fake(*args, **attrs)
    weight_mode = args[13] if len(args) > 13 else attrs["w_qr_quant_mode"]
    if weight_mode == WQrQuantMode.A8W8_MXFP8.value:
        descale_qr = descale_qr.view(_torch.uint8)
    return (q, descale_q, qr, descale_qr)


def quant_mla_prolog(
    x: _torch.Tensor,
    wqa: _torch.Tensor,
    wqb: _torch.Tensor,
    wkva: _torch.Tensor,
    wkb: _torch.Tensor,
    descale_x: _torch.Tensor,
    descale_wqa: _torch.Tensor,
    descale_wqb: _torch.Tensor,
    descale_wkva: _torch.Tensor,
    norm_weight_qa: _torch.Tensor,
    norm_weight_kva: _torch.Tensor,
    kv_cache: _torch.Tensor,
    cache_index: _torch.Tensor,
    w_qr_quant_mode: WQrQuantMode,
    qkv_quant_mode: QkvQuantMode,
    *,
    quant_scale_kv: _torch.Tensor | None = None,
    rope_sin: _torch.Tensor | None = None,
    rope_cos: _torch.Tensor | None = None,
    norm_eps: float = 1e-05,
    cache_mode: CacheMode = CacheMode.PA_BSND,
    qr_scale: float = 1.0,
    kvr_scale: float = 1.0,
    rope_mode: RopeMode = RopeMode.ROPE_INTERLEAVE_HALF,
    return_qr: bool = False,
) -> tuple[_torch.Tensor, _torch.Tensor, _torch.Tensor, _torch.Tensor]:
    """Run quantized MLA projection, normalization, RoPE and cache update.

    ``x``, weights and their descales use the selected MXFP8/HiF8 layout.
    FP32 normalization weights and
    ``cache_index`` describe normalization and the slots updated in ``kv_cache``.
    Optional scale and RoPE tensors are keyword arguments. Mode attributes use
    the enums attached to this function; scalar attributes control epsilon and
    output scaling. ``precheck`` validates scale contents before graph capture.

    Returns ``(q, descale_q, qr, descale_qr)``; the QR slots are empty when
    ``return_qr`` is false. Cache is updated in place and is not returned.
    """
    if type(return_qr) is not bool:
        raise TypeError("return_qr must be bool")
    _validate_scalars(norm_eps, qr_scale, kvr_scale)
    w_qr_quant_mode = _mode_value("w_qr_quant_mode", w_qr_quant_mode, WQrQuantMode)
    qkv_quant_mode = _mode_value("qkv_quant_mode", qkv_quant_mode, QkvQuantMode)
    cache_mode = _mode_value("cache_mode", cache_mode, CacheMode)
    rope_mode = _mode_value("rope_mode", rope_mode, RopeMode)
    _validate_modes(w_qr_quant_mode, qkv_quant_mode, cache_mode, rope_mode)
    graph_execution = _torch.compiler.is_compiling() or (
        isinstance(x, _torch.Tensor)
        and x.device.type in ("npu", "privateuseone")
        and (x.shape[0] != 0)
        and _torch.npu.is_current_stream_capturing()
    )
    if graph_execution and w_qr_quant_mode == WQrQuantMode.A8W8_MXFP8.value:
        if not isinstance(descale_x, _torch.Tensor) or descale_x.dtype != _E8:
            raise TypeError("descale_x must be an E8M0 Tensor")
        for name, value, weight in (
            ("descale_wqa", descale_wqa, wqa),
            ("descale_wqb", descale_wqb, wqb),
            ("descale_wkva", descale_wkva, wkva),
        ):
            if not isinstance(value, _torch.Tensor):
                raise TypeError(f"{name} must be a Tensor")
            byte_weight_scale = value.dtype == _torch.uint8 and tuple(value.shape) == (
                weight.shape[0] // 64,
                weight.shape[1],
                2,
            )
            if value.dtype != _E8 and (not byte_weight_scale):
                raise TypeError(f"{name} must be E8M0 or a UINT8 NZ weight scale")
        (descale_x, descale_wqa, descale_wqb, descale_wkva) = (
            value.view(_torch.uint8)
            for value in (descale_x, descale_wqa, descale_wqb, descale_wkva)
        )
    args = (
        x,
        wqa,
        wqb,
        wkva,
        wkb,
        descale_x,
        descale_wqa,
        descale_wqb,
        descale_wkva,
        norm_weight_qa,
        norm_weight_kva,
        kv_cache,
        cache_index,
    )
    attrs = dict(
        quant_scale_kv=quant_scale_kv,
        rope_sin=rope_sin,
        rope_cos=rope_cos,
        norm_eps=norm_eps,
        w_qr_quant_mode=w_qr_quant_mode,
        qkv_quant_mode=qkv_quant_mode,
        cache_mode=cache_mode,
        qr_scale=qr_scale,
        kvr_scale=kvr_scale,
        rope_mode=rope_mode,
        return_qr=return_qr,
    )
    dispatch = (
        _torch.ops.cann_ops_transformer._quant_mla_prolog_graph.default
        if graph_execution
        else _torch.ops.cann_ops_transformer.quant_mla_prolog.default
    )
    result = dispatch(*args, **attrs)
    if graph_execution and w_qr_quant_mode == WQrQuantMode.A8W8_MXFP8.value:
        return (result[0], result[1], result[2], result[3].view(_E8))
    return result


get_as_library().define(_TORCH_SCHEMA)
get_as_library().define(
    _TORCH_SCHEMA.replace("quant_mla_prolog(", "_quant_mla_prolog_graph(", 1)
)
impl(get_as_library(), "quant_mla_prolog", "Meta")(_fake)
impl(get_as_library(), "_quant_mla_prolog_graph", "Meta")(_graph_fake)


@impl(get_as_library(), "quant_mla_prolog", "PrivateUse1")
def _dispatch_eager(*values, **attributes):
    from .op_dsl import quant_mla_prolog as implementation

    return implementation.execute_async(*values, **attributes)


@impl(get_as_library(), "_quant_mla_prolog_graph", "PrivateUse1")
def _dispatch_graph(*values, **attributes):
    from .op_dsl import quant_mla_prolog as implementation

    return implementation.execute_graph(*values, **attributes)


def _mode(name, value, enum_type, implementation):
    if type(value) is not enum_type:
        raise TypeError(f"{name} must be {enum_type.__name__}")
    return getattr(implementation, enum_type.__name__)(value.value)


def quant_mla_prolog_precheck(
    x: _torch.Tensor,
    wqa: _torch.Tensor,
    wqb: _torch.Tensor,
    wkva: _torch.Tensor,
    wkb: _torch.Tensor,
    descale_x: _torch.Tensor,
    descale_wqa: _torch.Tensor,
    descale_wqb: _torch.Tensor,
    descale_wkva: _torch.Tensor,
    norm_weight_qa: _torch.Tensor,
    norm_weight_kva: _torch.Tensor,
    kv_cache: _torch.Tensor,
    cache_index: _torch.Tensor,
    w_qr_quant_mode: WQrQuantMode,
    qkv_quant_mode: QkvQuantMode,
    *,
    quant_scale_kv: _torch.Tensor | None = None,
    rope_sin: _torch.Tensor | None = None,
    rope_cos: _torch.Tensor | None = None,
    norm_eps: float = 1e-05,
    cache_mode: CacheMode = CacheMode.PA_BSND,
    qr_scale: float = 1.0,
    kvr_scale: float = 1.0,
    rope_mode: RopeMode = RopeMode.ROPE_INTERLEAVE_HALF,
    return_qr: bool = False,
) -> None:
    from .op_dsl import quant_mla_prolog as implementation

    return implementation.quant_mla_prolog_precheck(
        x,
        wqa,
        wqb,
        wkva,
        wkb,
        descale_x,
        descale_wqa,
        descale_wqb,
        descale_wkva,
        norm_weight_qa,
        norm_weight_kva,
        kv_cache,
        cache_index,
        quant_scale_kv=quant_scale_kv,
        rope_sin=rope_sin,
        rope_cos=rope_cos,
        norm_eps=norm_eps,
        w_qr_quant_mode=_mode(
            "w_qr_quant_mode", w_qr_quant_mode, WQrQuantMode, implementation
        ),
        qkv_quant_mode=_mode(
            "qkv_quant_mode", qkv_quant_mode, QkvQuantMode, implementation
        ),
        cache_mode=_mode("cache_mode", cache_mode, CacheMode, implementation),
        qr_scale=qr_scale,
        kvr_scale=kvr_scale,
        rope_mode=_mode("rope_mode", rope_mode, RopeMode, implementation),
        return_qr=return_qr,
    )


impl(get_as_library(), "quant_mla_prolog", "Functionalize")(_reject_functionalization)
quant_mla_prolog.QkvQuantMode = QkvQuantMode
quant_mla_prolog.CacheMode = CacheMode
quant_mla_prolog.RopeMode = RopeMode
quant_mla_prolog.WQrQuantMode = WQrQuantMode
quant_mla_prolog.precheck = quant_mla_prolog_precheck
