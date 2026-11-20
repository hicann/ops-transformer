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

"""MLA v4 的 TTK 入口适配，保留 CSV 槽位顺序和属性，在接口边界转换公开枚举。"""

import importlib.util
import math
import os
import sys

import torch


def _transport():
    name = "mla_prolog_v4_ttk_assets_impl_transport"
    if name not in sys.modules:
        path = os.path.join(os.path.dirname(__file__), "impl", "transport.py")
        spec = importlib.util.spec_from_file_location(name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


# 只加载当前仓库中该算子的 op_dsl 源码。
def _resolve_sample_file():
    path = os.path.join(
        os.path.dirname(os.path.dirname(_THIS_DIR)), "op_dsl", "mla_prolog_v4.py"
    )
    if not os.path.isfile(path):
        raise ImportError(f"operator source not found: {path}")
    return path


_UNDER_TEST_MODULE = "_mla_prolog_v4_under_test"


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


# 模式属性映射为整数值，CSV 可使用整数或受支持的枚举成员名称。

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
_ASSET_KWARGS = frozenset({"cache_index_rule"})


def _validate_unknown_attributes(kwargs, allowed=_ASSET_KWARGS):
    """拒绝没有消费者的属性名，避免拼写错误静默使用默认值。"""
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


def _validate_weight_format(weight_format):
    """严格解析 weight_format，只接受 ND/NZ。"""
    key = str(weight_format).strip().upper()
    if key in ("NZ", "FRACTAL_NZ", "29"):
        return True
    if key in ("ND", "0", "2"):
        return False
    raise ValueError(
        f"unsupported weight_format: {weight_format!r} (expected 'ND' or 'NZ')"
    )


_MODE_NAMES = {
    "qkv_quant_mode": {
        "A16C16": 0,
        "NO_QUANT": 0,
        "A16C8_KV_NOPE_FP8_E4M3_PER_TOKEN_GROUP_GROUP_SIZE_128_ROPE_BF16_SCALE_FLOAT_KV_CACHE_LAYOUT_N_NOPE_ROPE_SCALE": 1,
        "KV_FP8_PER_TOKEN_GROUP": 1,
        "A8C8_Q_FP8_E4M3_PER_TOKEN_HEAD_KV_FP8_E4M3_PER_TENSOR": 2,
        "Q_KV_FP8": 2,
        "A8C8_QKV_HIFLOAT8_PER_TENSOR": 3,
        "Q_KV_HIF8": 3,
    },
    "cache_mode": {"PA_BSND": 0, "PA_NZ": 1},
    "rope_mode": {
        "NO_ROPE": 0,
        "NONE": 0,
        "ROPE_INTERLEAVE_HALF": 1,
        "INTERLEAVE_HALF": 1,
        "ROPE_HALF": 2,
        "HALF": 2,
    },
}


def _finite_scalar(name, value, positive=False):
    """校验实数标量，拒绝 bool、非标量、非有限值和 FP32 溢出。"""
    if not (isinstance(value, (int, float)) and not isinstance(value, bool)):
        raise ValueError(f"{name} must be a real scalar")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    scalar = torch.tensor(value, dtype=torch.float32).item()
    if not math.isfinite(scalar):
        raise ValueError(f"{name} must be finite in FP32")
    if positive and not scalar > 0.0:
        raise ValueError(f"{name} must be positive in FP32")
    return scalar


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


def _validate_attributes(
    norm_eps, qr_scale, kvr_scale, qkv_quant_mode, cache_mode, rope_mode, return_qr
):
    """执行公开属性契约检查。

    返回规范化的 (norm_eps, qr_scale, kvr_scale, qkv, cache, rope)。
    """
    norm_eps = _finite_scalar("norm_eps", norm_eps, positive=True)
    qr_scale = _finite_scalar("qr_scale", qr_scale)
    kvr_scale = _finite_scalar("kvr_scale", kvr_scale)
    qkv = _mode_int("qkv_quant_mode", qkv_quant_mode)
    if qkv not in (0, 1, 2, 3):
        raise ValueError("qkv_quant_mode must be 0/1/2/3")
    rope = _mode_int("rope_mode", rope_mode)
    if rope not in (0, 1, 2):
        raise ValueError("rope_mode must be 0/1/2")
    cache = _mode_int("cache_mode", cache_mode)
    if cache not in (0, 1):
        raise ValueError("cache_mode must be 0(PA_BSND) or 1(PA_NZ)")
    if type(return_qr) is not bool:
        raise ValueError("return_qr must be bool")
    if qkv == 1 and cache == 1:
        raise ValueError("mode 1 supports PA_BSND only")
    return norm_eps, qr_scale, kvr_scale, qkv, cache, rope


def _cache_mode(sample, cache_mode):
    """将 CSV 的 cache_mode 映射到公开 CacheMode 枚举。

    接受 PA_BSND/PA_NZ 名称、整数值或数字字符串。
    两种布局均在相同物理字节上由设备与 CPU 参考原地更新，保持逐输出比较对应。
    """
    if isinstance(cache_mode, str):
        key = cache_mode.strip().upper()
        if key == "PA_BSND":
            return sample.CacheMode.PA_BSND
        if key == "PA_NZ":
            return sample.CacheMode.PA_NZ
        raise ValueError(f"unsupported cache_mode: {cache_mode}")
    return sample.CacheMode(int(cache_mode))


def mla_prolog_v4_ttk(
    x: torch.Tensor,
    wqa: torch.Tensor,
    wqb: torch.Tensor,
    wkva: torch.Tensor,
    wkb: torch.Tensor,
    norm_weight_qa: torch.Tensor,
    norm_weight_kva: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_index: torch.Tensor,
    quant_scale_kv: torch.Tensor = None,
    rope_sin: torch.Tensor = None,
    rope_cos: torch.Tensor = None,
    *,
    norm_eps: float = 1e-5,
    qkv_quant_mode: int = 0,
    cache_mode: str = "PA_BSND",
    qr_scale: float = 1.0,
    kvr_scale: float = 1.0,
    rope_mode: int = 1,
    return_qr: bool = False,
    weight_format: str = None,
    **kwargs,
):
    """mla_prolog_v4_ttk(x: Tensor, wqa: Tensor, wqb: Tensor, wkva: Tensor, wkb: Tensor, norm_weight_qa: Tensor, norm_weight_kva: Tensor, kv_cache: Tensor, cache_index: Tensor, quant_scale_kv: Tensor=None, rope_sin: Tensor=None, rope_cos: Tensor=None, *, norm_eps: float=1e-05, qkv_quant_mode: int=0, cache_mode: int='PA_BSND', qr_scale: float=1.0, kvr_scale: float=1.0, rope_mode: int=1, return_qr: bool=False, weight_format: str=None)

    执行一次公开算子调用，返回四个与 Golden 对应的输出槽位。

    张量位置参数保持 CSV 槽位顺序，控制属性来自 attributes，权重布局来自 tensor_formats。
    quant_scale_kv 及 RoPE 张量按模式可选，kv_cache 由算子原地更新。
    返回 [q, descale_q, qr, kv_cache_updated]；量化槽位由比较钩子按对应数值契约验收。

    Args:
        qkv_quant_mode (float or str): 保留原始数值类型，仅接受严格整数或受支持的模式名称。
        cache_mode (float or str): 保留原始数值类型，仅接受严格整数或受支持的模式名称。
        rope_mode (float or str): 保留原始数值类型，仅接受严格整数或受支持的模式名称。
    """
    (
        x,
        wqa,
        wqb,
        wkva,
        wkb,
        norm_weight_qa,
        norm_weight_kva,
        kv_cache,
        cache_index,
        quant_scale_kv,
        rope_sin,
        rope_cos,
    ) = [
        _transport().to_torch(v)
        for v in (
            x,
            wqa,
            wqb,
            wkva,
            wkb,
            norm_weight_qa,
            norm_weight_kva,
            kv_cache,
            cache_index,
            quant_scale_kv,
            rope_sin,
            rope_cos,
        )
    ]
    _validate_unknown_attributes(kwargs)
    weight_format = _resolve_weight_format(
        (wqa, wqb, wkva), kwargs.get("tensor_formats"), weight_format
    )
    _validate_weight_format(weight_format)
    sample = _load_under_test()
    norm_eps, qr_scale, kvr_scale, qkv_mode, cache_mode, rope_mode = (
        _validate_attributes(
            norm_eps,
            qr_scale,
            kvr_scale,
            qkv_quant_mode,
            cache_mode,
            rope_mode,
            return_qr,
        )
    )
    q, descale_q, qr = sample.mla_prolog_v2(
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
        qkv_quant_mode=sample.QkvQuantMode(qkv_mode),
        cache_mode=_cache_mode(sample, cache_mode),
        qr_scale=qr_scale,
        kvr_scale=kvr_scale,
        rope_mode=sample.RopeMode(rope_mode),
        # return_qr 原样传递，由严格 bool 契约拒绝整数 1 和字符串 False 等输入。
        return_qr=return_qr,
    )
    torch.npu.synchronize()
    # Cache 原地更新；两种布局均保留输入字节顺序，与 CPU Golden 逐元素对应。

    return [q, descale_q, qr, kv_cache]
