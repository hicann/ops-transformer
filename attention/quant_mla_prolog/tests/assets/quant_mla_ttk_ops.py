#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Quant MLA 的 TTK 入口适配，保留 CSV 槽位顺序和属性，在接口边界转换公开枚举。"""

import importlib.util
import math
import os
import struct
import sys

import numpy as np
import torch


def _transport():
    name = "quant_mla_prolog_ttk_assets_impl_transport"
    if name not in sys.modules:
        path = os.path.join(os.path.dirname(__file__), "impl", "transport.py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# 固定加载当前仓库中的算子源码。
def _resolve_sample_file():
    path = os.path.join(
        os.path.dirname(os.path.dirname(_THIS_DIR)), "op_dsl", "quant_mla_prolog.py"
    )
    if not os.path.isfile(path):
        raise ImportError(f"operator source not found: {path}")
    return path


_UNDER_TEST_MODULE = "_quant_mla_prolog_under_test"

# 模式属性名称映射为成员名称，接受 CSV 的枚举名或原始整数。

_MODE_NAMES = {
    "w_qr_quant_mode": {
        "MXFP8": 1,
        "A8W8_MXFP8": 1,
        "HIF8": 2,
        "A8W8_HIFLOAT8_PER_TENSOR": 2,
        "HIFLOAT4": 3,
        "A4W4_HIFLOAT4": 3,
    },
    "qkv_quant_mode": {
        "NO_QUANT": 0,
        "A16C16": 0,
        "KV_FP8_PER_TOKEN_GROUP": 1,
        "A16C8_KV_NOPE_FP8_E4M3_PER_TOKEN_GROUP_GROUP_SIZE_128_ROPE_BF16_SCALE_FLOAT_KV_CACHE_LAYOUT_N_NOPE_ROPE_SCALE": 1,
        "Q_KV_FP8": 2,
        "A8C8_Q_FP8_E4M3_PER_TOKEN_HEAD_KV_FP8_E4M3_PER_TENSOR": 2,
        "Q_KV_HIF8": 3,
        "A8C8_QKV_HIFLOAT8_PER_TENSOR": 3,
    },
    "cache_mode": {"PA_BSND": 0, "PA_NZ": 1},
    "rope_mode": {
        "NONE": 0,
        "NO_ROPE": 0,
        "INTERLEAVE_HALF": 1,
        "ROPE_INTERLEAVE_HALF": 1,
        "HALF": 2,
        "ROPE_HALF": 2,
    },
}

# 按模式名称获取公开枚举，调用入口前将原始整数映射为对应枚举成员。

_ENUM_ATTRS = {
    "w_qr_quant_mode": "WQrQuantMode",
    "qkv_quant_mode": "QkvQuantMode",
    "cache_mode": "CacheMode",
    "rope_mode": "RopeMode",
}


def _load_under_test():
    """每个进程只加载一次被测算子模块。

    模块导入会注册 torch.library schema；使用固定 sys.modules 键避免以不同模块名重复注册。
    """
    module = sys.modules.get(_UNDER_TEST_MODULE)
    if module is not None:
        return module
    sample_file = _resolve_sample_file()
    import torch_npu  # noqa: F401  # 注册算子模块需要的 torch.npu

    spec = importlib.util.spec_from_file_location(_UNDER_TEST_MODULE, sample_file)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_UNDER_TEST_MODULE] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(_UNDER_TEST_MODULE, None)
        raise
    return module


# 公开接口支持的 (qkv_quant_mode,w_qr_quant_mode) 组合。
_VALID_MODE_PAIRS = {(0, 1), (0, 2), (1, 1), (2, 1), (3, 2)}


def _mode_int(name, value):
    """解析 CSV 整数或模式名称，不截断数值，也不接受其他类型的枚举。

    枚举遵循公开接口契约；CSV 名称是显式传输形式，不作为任意数字字符串处理。
    """
    from enum import Enum

    if type(value) is int:
        return value
    if isinstance(value, Enum):
        sample = _load_under_test()
        enum_names = {
            "qkv_quant_mode": "QkvQuantMode",
            "cache_mode": "CacheMode",
            "rope_mode": "RopeMode",
            "w_qr_quant_mode": "WQrQuantMode",
        }
        expected = getattr(sample, enum_names[name])
        if type(value) is not expected:
            raise TypeError(f"{name} must use the operator's {expected.__name__} enum")
        return value.value
    if type(value) is str:
        key = value.strip().upper()
        if key in _MODE_NAMES.get(name, {}):
            return _MODE_NAMES[name][key]
    raise TypeError(
        f"{name} must be a plain integer, a named CSV mode, or its public enum"
    )


def _validate_scalars(norm_eps, qr_scale, kvr_scale):
    """校验标量契约，拒绝 bool，并保证转换为 FP32 后仍为有限值。"""
    if type(norm_eps) not in (float, int):
        raise TypeError("norm_eps must be a Python float or int, not bool or Tensor")
    try:
        fp32_eps = struct.unpack("f", struct.pack("f", norm_eps))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError(
            "norm_eps is not representable as positive finite FP32"
        ) from error
    if not (math.isfinite(fp32_eps) and fp32_eps > 0):
        raise ValueError("norm_eps must be finite and strictly positive")
    for name, value in (("qr_scale", qr_scale), ("kvr_scale", kvr_scale)):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a Python float or int, not bool or Tensor")
        try:
            fp32_value = struct.unpack("f", struct.pack("f", float(value)))[0]
        except (OverflowError, struct.error) as error:
            raise ValueError(f"{name} must be finite in FP32") from error
        if not math.isfinite(fp32_value):
            raise ValueError(f"{name} must be finite in FP32")


def _validate_modes(qkv_quant_mode, w_qr_quant_mode, cache_mode, rope_mode, return_qr):
    """按公开算子的模式契约校验输入，保留对应异常消息。"""
    qkv = _mode_int("qkv_quant_mode", qkv_quant_mode)
    w_qr = _mode_int("w_qr_quant_mode", w_qr_quant_mode)
    cache = _mode_int("cache_mode", cache_mode)
    rope = _mode_int("rope_mode", rope_mode)
    if (qkv, w_qr) not in _VALID_MODE_PAIRS:
        raise ValueError(f"unsupported (qkv_quant_mode,w_qr_quant_mode)=({qkv},{w_qr})")
    if cache not in (0, 1):
        raise ValueError("cache_mode must be 0(PA_BSND) or 1(PA_NZ)")
    if qkv == 1 and cache == 1:
        raise ValueError("qkv_quant_mode=1 does not support PA_NZ")
    if rope not in (0, 1, 2):
        raise ValueError("rope_mode must be 0, 1, or 2")
    if type(return_qr) is not bool:
        raise TypeError("return_qr must be bool")


def _mode_value(sample, name, value):
    """将模式值映射为公开算子的枚举成员。

    接受受支持的成员名称；若模块没有相应枚举，则原样返回已经解析的整数。
    """
    resolved = _mode_int(name, value)
    enum_type = getattr(sample, _ENUM_ATTRS[name], None)
    if enum_type is None:
        return resolved
    return enum_type(resolved)


def _as_e8m0(tensor):
    """将单字节 E8M0 原始载体查看为原生 E8M0 dtype。

    Tensor.view 保留字节且不复制，因此不改变数值或 CPU Golden。
    已有原生 E8M0 张量直接传递；环境没有 torch.float8_e8m0fnu 时也不转换。
    CSV 使用原生 E8M0 NumPy dtype 时需要 en-dtypes，设备接口要求对应的原生 dtype。
    """
    if tensor is None or not torch.is_tensor(tensor):
        return tensor
    native = getattr(torch, "float8_e8m0fnu", None)
    if native is None or tensor.dtype == native:
        return tensor
    if _device_format(tensor) == _NZ_FORMAT:
        # 已有 format-29 的 MX scale 保留 uint8 载体，避免 dtype 视图丢失格式描述。

        return tensor
    if tensor.dtype in (torch.uint8, torch.int8):
        return tensor.view(native)
    return tensor


def _pack_nz_descale(scale, byte_carrier=False):
    """构造 FRACTAL_NZ 的 MX 权重 descale。

    逻辑 ND 形状为 (O,K//64,2)，设备 format-29 的逻辑形状为 (K//64,O,2)，
    物理形状为 (O/16,K//64,16,2)。先 reshape/permute 打包，再构造逻辑视图并转换设备格式。
    byte_carrier 路径以 uint8 传递字节，同时指定 E8M0 input_dtype。
    """
    if scale is None or not torch.is_tensor(scale) or scale.dim() != 3:
        return scale
    width, groups, pair = scale.shape
    if width % 16:
        raise ValueError(f"NZ descale needs width%16==0, got {tuple(scale.shape)}")
    import torch_npu  # 延迟导入，使没有 torch_npu 的主机也能解析 CSV。

    raw = scale.view(torch.uint8) if scale.dtype != torch.uint8 else scale
    packed = raw.reshape(width // 16, 16, groups, pair).permute(0, 2, 1, 3).contiguous()
    logical = packed.permute(1, 0, 2, 3).contiguous().reshape(groups, width, pair)
    native = getattr(torch, "float8_e8m0fnu", None)
    if byte_carrier or native is None:
        return (
            torch_npu.npu_format_cast(logical.npu(), 29, input_dtype=native)
            if native is not None
            else logical.npu()
        )
    return torch_npu.npu_format_cast(logical.view(native).npu(), 29)


_TTK_INJECTED_KWARGS = frozenset(
    {
        "device",
        "backend",
        "golden_mode",
        "tensor_formats",
        "tensor_dtypes",
        "use_numpy",
        "short_soc_version",
        "testcase_name",
        "scalar_dtypes",
        "input_ranges",
        "batch_axis",
        "batch_slice_info",
        "batch_seed",
    }
)
_ASSET_KWARGS = frozenset({"cache_index_rule", "hif8_input_style", "e8m0_input_style"})


def _validate_unknown_attributes(kwargs, allowed=_ASSET_KWARGS):
    """拒绝没有消费者的属性名，避免拼写错误静默使用默认值。

    TTK 会将未匹配的 CSV 属性和框架元数据传入 **kwargs，不能直接丢弃未知属性。
    """
    unknown = sorted(
        name
        for name in kwargs
        if name not in allowed
        and name not in _TTK_INJECTED_KWARGS
        and not name.startswith("batch_")
    )
    if unknown:
        raise TypeError(f"unexpected attribute(s): {', '.join(unknown)}")


def _resolve_weight_format(weights, tensor_formats=None, legacy=None):
    """CPU 端依据 CSV 格式声明，设备端依据张量实际格式。

    兼容属性只能核对格式声明，不能覆盖声明。
    """
    import ast

    def normalize(value):
        key = str(value).strip().upper()
        if key in ("ND", "0", "2"):
            return "ND"
        if key in ("NZ", "FRACTAL_NZ", "29"):
            return "NZ"
        raise ValueError(f"unsupported weight tensor format: {value!r}")

    declared = None
    if isinstance(tensor_formats, str):
        tensor_formats = ast.literal_eval(tensor_formats)
    if tensor_formats is not None:
        if len(tensor_formats) < 4:
            raise ValueError("incomplete tensor_formats for wqa/wqb/wkva")
        formats = [normalize(tensor_formats[i]) for i in (1, 2, 3)]
        if len(set(formats)) != 1:
            raise ValueError(
                "wqa/wqb/wkva tensor_formats must be uniformly ND or FRACTAL_NZ"
            )
        declared = formats[0]
    actual = []
    for weight in weights:
        if weight is not None and str(getattr(weight, "device", "")).startswith("npu"):
            import torch_npu

            actual.append(normalize(torch_npu.get_npu_format(weight)))
    if actual:
        if len(actual) != 3 or len(set(actual)) != 1:
            raise ValueError("wqa/wqb/wkva actual formats must be uniform")
        if declared is not None and declared != actual[0]:
            raise ValueError("actual weight formats differ from tensor_formats")
        declared = actual[0]
    if legacy is not None:
        previous = normalize(legacy)
        if declared is not None and previous != declared:
            raise ValueError("legacy weight_format conflicts with tensor_formats")
        if declared is None:
            declared = previous
    if declared is None:
        raise ValueError(
            "weight format requires tensor_formats or actual device tensors"
        )
    return declared


def _is_nz(weight_format):
    """严格解析 weight_format，只接受 ND/NZ。"""
    key = str(weight_format).strip().upper()
    if key in ("NZ", "FRACTAL_NZ", "29"):
        return True
    if key in ("ND", "0", "2"):
        return False
    raise ValueError(
        f"unsupported weight_format: {weight_format!r} (expected 'ND' or 'NZ')"
    )


_NZ_FORMAT = 29
_ND_FORMATS = (0, 2)


def _device_format(tensor):
    """返回设备张量的实际 NPU 格式；主机张量或缺少 torch_npu 时返回 None。"""
    if tensor is None or not torch.is_tensor(tensor):
        return None
    if tensor.device.type != "npu":
        return None
    try:
        import torch_npu  # 按需导入设备依赖。

        return int(torch_npu.get_npu_format(tensor))
    except Exception:  # noqa: BLE001  # 主机张量没有设备格式
        return None


def _validate_device_formats(weights, descales, descale_x, mode, weight_format):
    """只校验张量实际设备格式，不执行格式转换。

    MXFP8 的 NZ 权重要求 format-29 权重 descale，ND 权重要求 ND descale，descale_x 始终为 ND。
    HiF8 权重可为 ND/NZ，但 FP32 权重 descale 与 descale_x 均要求 ND。
    """
    actual = [_device_format(w) for w in weights]
    if any(fmt is None for fmt in actual):
        return  # 主机张量或缺少 torch_npu 时不查询设备格式。
    configured = _is_nz(weight_format)
    expects_nz = _NZ_FORMAT if configured else _ND_FORMATS
    ok = lambda fmt: fmt == _NZ_FORMAT if configured else fmt in _ND_FORMATS  # noqa: E731
    if not all(ok(fmt) for fmt in actual) or len(set(actual)) != 1:
        raise ValueError(
            "wqa/wqb/wkva actual formats must be uniformly "
            f"{'FRACTAL_NZ (29)' if configured else 'ND'} to match weight_format="
            f"{weight_format!r}, got {actual}"
        )
    if mode == 1:
        for name, scale in zip(
            ("descale_wqa", "descale_wqb", "descale_wkva"), descales
        ):
            fmt = _device_format(scale)
            if fmt is None:
                continue
            if not ok(fmt):
                raise ValueError(
                    f"{name} must match weight format: "
                    f"{'FRACTAL_NZ (29)' if configured else 'ND'}, got format {fmt}"
                )
        x_fmt = _device_format(descale_x)
        if x_fmt is not None and x_fmt not in _ND_FORMATS:
            raise ValueError(f"descale_x must be ND, got format {x_fmt}")
    else:
        for name, scale in zip(
            ("descale_wqa", "descale_wqb", "descale_wkva"), descales
        ):
            fmt = _device_format(scale)
            if fmt is not None and fmt not in _ND_FORMATS:
                raise ValueError(f"{name} must stay ND for HiF8, got format {fmt}")
        x_fmt = _device_format(descale_x)
        if x_fmt is not None and x_fmt not in _ND_FORMATS:
            raise ValueError(f"descale_x must be ND, got format {x_fmt}")
    return expects_nz


def _invoke(
    sample,
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
    quant_scale_kv,
    rope_sin,
    rope_cos,
    *,
    norm_eps,
    w_qr_quant_mode,
    qkv_quant_mode,
    cache_mode,
    qr_scale,
    kvr_scale,
    rope_mode,
    return_qr,
    weight_format="ND",
    sync=True,
    check_formats=True,
):
    """解析 CSV 属性后调用公开算子入口，不直接调用 Dispatcher 名称。

    公开入口识别图执行，MXFP8 在 torch.compile 下将 E8M0 descale 查看为 uint8。
    该视图使 npugraph_ex 能分解 auto_functionalized_v2，避免直接传原生 E8M0 导致分解断言失败。
    """
    # 在设备操作前拒绝非法属性，遵循公开接口契约，不静默将 True 转为 1.0。

    _validate_scalars(norm_eps, qr_scale, kvr_scale)
    _validate_modes(qkv_quant_mode, w_qr_quant_mode, cache_mode, rope_mode, return_qr)
    # CSV 声明 descale 的 NZ 布局 (K//64,O,2)，TTK 绑定时完成 format-29 转换；此处只校验实际格式。

    _is_nz(weight_format)  # 非法值直接报错，合法声明仅用于后续实际格式核对。
    if check_formats:
        _validate_device_formats(
            (wqa, wqb, wkva),
            (descale_wqa, descale_wqb, descale_wkva),
            descale_x,
            _mode_int("w_qr_quant_mode", w_qr_quant_mode),
            weight_format,
        )
    q, descale_q, qr, descale_qr = sample.quant_mla_prolog(
        x,
        wqa,
        wqb,
        wkva,
        wkb,
        _as_e8m0(descale_x),
        _as_e8m0(descale_wqa),
        _as_e8m0(descale_wqb),
        _as_e8m0(descale_wkva),
        norm_weight_qa,
        norm_weight_kva,
        kv_cache,
        cache_index,
        quant_scale_kv=quant_scale_kv,
        rope_sin=rope_sin,
        rope_cos=rope_cos,
        norm_eps=norm_eps,
        w_qr_quant_mode=_mode_value(sample, "w_qr_quant_mode", w_qr_quant_mode),
        qkv_quant_mode=_mode_value(sample, "qkv_quant_mode", qkv_quant_mode),
        cache_mode=_mode_value(sample, "cache_mode", cache_mode),
        qr_scale=qr_scale,
        kvr_scale=kvr_scale,
        rope_mode=_mode_value(sample, "rope_mode", rope_mode),
        # return_qr 原样传递，由严格 bool 契约拒绝整数等非法输入。
        return_qr=return_qr,
    )
    if sync:
        torch.npu.synchronize()
    return q, descale_q, qr, descale_qr


def _ensure_torch(value):
    """将 NumPy 输入适配为 Torch 张量，语义 HiF8 保留原始字节。

    出现非 Torch 原生 dtype 时，TTK 的 NumPy 回退路径会将全部输入表示为 ndarray。
    HiF8 无对应 Torch dtype 时，通过 uint8 视图传递，遵循公开接口的 uint8/int8 载体契约。
    """
    if value is None:
        return value
    if torch.is_tensor(value):
        # 原生 HiF8 以 uint8/int8 单字节载体暴露给公开接口，不进行数值转换。

        return value.view(torch.uint8) if "hifloat" in str(value.dtype) else value
    arr = np.asarray(value)
    name = str(arr.dtype)
    if "hifloat8" in name or name in ("uint8", "int8") and arr.dtype.itemsize == 1:
        if name in ("uint8", "int8"):
            return torch.from_numpy(arr).view(torch.uint8)
        return torch.from_numpy(arr.view(np.uint8))
    if "bfloat16" in name:
        return torch.from_numpy(arr.view(np.uint16)).view(torch.bfloat16)
    if "float8_e4m3fn" in name:
        return torch.from_numpy(arr.view(np.uint8)).view(torch.float8_e4m3fn)
    if "float8_e5m2" in name:
        return torch.from_numpy(arr.view(np.uint8)).view(torch.float8_e5m2)
    if "e8m0" in name:
        native = getattr(torch, "float8_e8m0fnu", None)
        raw = torch.from_numpy(arr.view(np.uint8))
        return raw.view(native) if native is not None else raw
    return torch.from_numpy(arr)


def quant_mla_prolog_ttk(
    x: torch.Tensor,
    wqa: torch.Tensor,
    wqb: torch.Tensor,
    wkva: torch.Tensor,
    wkb: torch.Tensor,
    descale_x: torch.Tensor,
    descale_wqa: torch.Tensor,
    descale_wqb: torch.Tensor,
    descale_wkva: torch.Tensor,
    norm_weight_qa: torch.Tensor,
    norm_weight_kva: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_index: torch.Tensor,
    quant_scale_kv: torch.Tensor = None,
    rope_sin: torch.Tensor = None,
    rope_cos: torch.Tensor = None,
    *,
    norm_eps: float = 1e-5,
    w_qr_quant_mode: int = 1,
    qkv_quant_mode: int = 0,
    cache_mode: int = 0,
    qr_scale: float = 1.0,
    kvr_scale: float = 1.0,
    rope_mode: int = 1,
    return_qr: bool = False,
    weight_format: str = None,
    **kwargs,
):
    """quant_mla_prolog_ttk(x: Tensor, wqa: Tensor, wqb: Tensor, wkva: Tensor, wkb: Tensor, descale_x: Tensor, descale_wqa: Tensor, descale_wqb: Tensor, descale_wkva: Tensor, norm_weight_qa: Tensor, norm_weight_kva: Tensor, kv_cache: Tensor, cache_index: Tensor, quant_scale_kv: Tensor=None, rope_sin: Tensor=None, rope_cos: Tensor=None, *, norm_eps: float=1e-05, w_qr_quant_mode: int=1, qkv_quant_mode: int=0, cache_mode: int=0, qr_scale: float=1.0, kvr_scale: float=1.0, rope_mode: int=1, return_qr: bool=False, weight_format: str=None)

    执行一次公开算子调用，返回五个与 Golden 对应的输出槽位。

    张量位置参数保持 CSV 槽位顺序，控制属性来自 attributes，权重布局来自 tensor_formats。
    quant_scale_kv 及 RoPE 张量按模式可选，kv_cache 由算子原地更新。
    返回 [q, descale_q, qr, descale_qr, kv_cache_updated]；量化槽位由比较钩子按对应数值契约验收。

    Args:
        w_qr_quant_mode (float or str): 保留原始数值类型，仅接受严格整数或受支持的模式名称。
        qkv_quant_mode (float or str): 保留原始数值类型，仅接受严格整数或受支持的模式名称。
        cache_mode (float or str): 保留原始数值类型，仅接受严格整数或受支持的模式名称。
        rope_mode (float or str): 保留原始数值类型，仅接受严格整数或受支持的模式名称。
    """
    _validate_unknown_attributes(kwargs)
    weight_format = _resolve_weight_format(
        (wqa, wqb, wkva), kwargs.get("tensor_formats"), weight_format
    )
    sample = _load_under_test()
    (
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
        quant_scale_kv,
        rope_sin,
        rope_cos,
    ) = (
        _ensure_torch(t)
        for t in (
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
            quant_scale_kv,
            rope_sin,
            rope_cos,
        )
    )
    q, descale_q, qr, descale_qr = _invoke(
        sample,
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
        quant_scale_kv,
        rope_sin,
        rope_cos,
        norm_eps=norm_eps,
        w_qr_quant_mode=w_qr_quant_mode,
        qkv_quant_mode=qkv_quant_mode,
        cache_mode=cache_mode,
        qr_scale=qr_scale,
        kvr_scale=kvr_scale,
        rope_mode=rope_mode,
        return_qr=return_qr,
        weight_format=weight_format,
        sync=True,
    )
    # Cache 原地更新；两种布局均保留输入字节顺序，与 CPU Golden 逐元素对应。

    return [q, descale_q, qr, descale_qr, kv_cache]
