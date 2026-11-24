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

"""MLA v4 的独立 CPU Golden 适配。

普通参考遵循单 BF16 数值契约，数值定义位于本地 pytest_reference.py，
不依赖 pytest 目录。三方比较使用独立的 FP32 数学参考。
Q 根据模式使用 BF16、E4M3FN 或 uint8 HiF8；descale_q 为 FP32，
QR 为 BF16 或空张量，Cache 保留各模式的存储类型。
模式 3 的语义 HiF8 输入仅按 uint8 字节查看，普通参考不将其数值转换为 BF16/FP32。
PA_NZ 保留物理打包布局。
"""

import importlib.util
import os
import sys

import torch

# 参考实现固定在当前 assets 目录，不依赖 pytest 侧源码。
_IMPL_DIR = os.path.dirname(os.path.abspath(__file__))
_ASSETS_DIR = os.path.dirname(_IMPL_DIR)
_REFERENCE_PATH = os.path.join(_IMPL_DIR, "golden_reference.py")
_REFERENCE_MODULE = "mla_prolog_v4_golden_reference"
_REQUIRED_ATTRS = ("CacheMode", "mla_prolog_v4")


def _load_reference():
    """每个进程仅加载一次本地双方比较参考。"""
    # 在计算大形状 CPU Golden 前设置 CPU 线程数。
    torch.set_num_threads(int(os.environ.get("MLA_PROLOG_CPU_THREADS", "8")))
    module = sys.modules.get(_REFERENCE_MODULE)
    if module is None:
        if not os.path.isfile(_REFERENCE_PATH):
            raise ImportError(f"golden reference not found: {_REFERENCE_PATH}")
        spec = importlib.util.spec_from_file_location(
            _REFERENCE_MODULE, _REFERENCE_PATH
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[_REFERENCE_MODULE] = module
        spec.loader.exec_module(module)
    missing = [name for name in _REQUIRED_ATTRS if not hasattr(module, name)]
    if missing:
        raise ImportError(
            f"golden reference {_REFERENCE_PATH!r} is not the expected mla_prolog_v4 "
            f"golden (missing {missing}); sync impl/golden_reference.py and keep the repository reference layout"
        )
    return module


def _as_bf16(tensor):
    """还原可选的 Golden 类型提升。

    提升到 FP32/FP64 不损失 BF16 值，转回可恢复原始 BF16 位模式；已有 BF16 输入直接传递。
    """
    if tensor is None or not torch.is_tensor(tensor):
        return tensor
    if tensor.dtype in (torch.float32, torch.float64):
        return tensor.to(torch.bfloat16)
    return tensor


def _golden_cache_mode(reference, cache_mode):
    """将 CSV 的 cache_mode 映射为参考模块的 CacheMode 枚举。

    PA_BSND 和 PA_NZ 均按算子的原地写入布局处理，使 CPU 与设备 Cache 保持对应。
    """
    if isinstance(cache_mode, str):
        key = cache_mode.strip().upper()
        if key == "PA_BSND":
            return reference.CacheMode.PA_BSND
        if key == "PA_NZ":
            return reference.CacheMode.PA_NZ
        return reference.CacheMode(int(key))
    if isinstance(cache_mode, bool):
        raise ValueError("cache_mode must be 0(PA_BSND) or 1(PA_NZ)")
    # 参考 CacheMode 是字符串枚举，因此通过名称映射整数模式。
    return (reference.CacheMode.PA_BSND, reference.CacheMode.PA_NZ)[int(cache_mode)]


def reference_weights(weights, weight_format):
    """将公开接口的 NZ (K,O) 权重映射为 CPU 参考使用的 (O,K)。"""
    if str(weight_format).strip().upper() in ("NZ", "FRACTAL_NZ", "29"):
        return tuple(weight.t().contiguous() for weight in weights)
    return tuple(weights)


def mla_prolog_v4_golden(
    x,
    wqa,
    wqb,
    wkva,
    wkb,
    norm_weight_qa,
    norm_weight_kva,
    kv_cache,
    cache_index,
    quant_scale_kv=None,
    rope_sin=None,
    rope_cos=None,
    *,
    norm_eps=1e-5,
    qkv_quant_mode=0,
    cache_mode="PA_BSND",
    qr_scale=1.0,
    kvr_scale=1.0,
    rope_mode=1,
    return_qr=False,
    weight_format=None,
    **kwargs,
):
    """使用与设备调用相同的参数规划执行 CPU Golden。"""
    # 遵循公开属性契约，拒绝布尔模式、非法标量及模式 1 与 PA_NZ 的组合，不静默强制转换。

    import mla_prolog_v4_ttk_ops as _adapter

    _adapter._validate_unknown_attributes(kwargs)
    weight_format = _adapter._resolve_weight_format(
        (wqa, wqb, wkva), kwargs.get("tensor_formats"), weight_format
    )
    _adapter._validate_weight_format(weight_format)
    norm_eps, qr_scale, kvr_scale, qkv_mode, cache_mode_id, rope_mode = (
        _adapter._validate_attributes(
            norm_eps,
            qr_scale,
            kvr_scale,
            qkv_quant_mode,
            cache_mode,
            rope_mode,
            return_qr,
        )
    )
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
        _adapter._transport().to_torch(v)
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
    reference = _load_reference()
    # 还原 TTK 可选的 BF16 类型提升，使参考计算使用声明的输入类型。

    x, wqa, wqb, wkva, wkb = (_as_bf16(t) for t in (x, wqa, wqb, wkva, wkb))
    wqa, wqb, wkva = reference_weights((wqa, wqb, wkva), weight_format)
    # 只有模式 0 的 Cache 使用 BF16，不转换量化载体的数值类型。
    if qkv_mode == 0:
        kv_cache = _as_bf16(kv_cache)
    rope_sin, rope_cos = (_as_bf16(t) for t in (rope_sin, rope_cos))
    # CPU Golden 使用独立的连续 Cache，隔离 CSV 底层存储及设备步长，避免原地写入影响设备输入。

    kv_cache = kv_cache.contiguous().clone()
    cache_mode_enum = _golden_cache_mode(reference, cache_mode)
    q_ref, descale_q_ref, qr_ref = reference.mla_prolog_v4(
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
        norm_eps=norm_eps,
        qkv_quant_mode=qkv_mode,
        cache_mode=cache_mode_enum,
        qr_scale=qr_scale,
        kvr_scale=kvr_scale,
        rope_mode=rope_mode,
        return_qr=return_qr,
    )
    # 参考在相同物理布局中原地更新 Cache，与包装器的第四个输出槽位对应。

    return [q_ref, descale_q_ref, qr_ref, kv_cache]
