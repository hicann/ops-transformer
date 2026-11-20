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

"""Quant MLA 的独立 CPU Golden 适配。

参考实现位于本地 golden_reference.py 和 pytest_reference.py，不依赖 pytest 目录。
参考按权重张量维度识别存储格式：二维为 ND，其余为 NZ。
输出槽位为 [q(T,N,512+Dr), descale_q, qr, descale_qr, kv_cache]。
算子与参考原地修改 Cache，适配层将其作为第五个槽位返回；
所有输出保留 cache_mode 选择的 PA_BSND/PA_NZ 物理布局。
"""

import importlib.util
import os
import sys

import numpy as np
import torch

_IMPL_DIR = os.path.dirname(os.path.abspath(__file__))
_REFERENCE_PATH = os.path.join(_IMPL_DIR, "golden_reference.py")
_REFERENCE_MODULE = "quant_mla_prolog_golden_reference"
_REQUIRED_ATTRS = ("quant_mla_prolog_golden", "cache_to_logical")


def _load_reference():
    """每个进程仅加载一次本地 CPU Golden。"""
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
            f"golden reference {_REFERENCE_PATH!r} is not the expected "
            f"quant_mla_prolog golden (missing {missing}); sync impl/golden_reference.py "
            "and keep the repository reference layout"
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


def _mode_int(value, names):
    import quant_mla_ttk_ops as adapter

    mappings = {
        "w_qr_quant_mode": _W_QR_QUANT_MODE,
        "qkv_quant_mode": _QKV_QUANT_MODE,
        "cache_mode": _CACHE_MODE,
        "rope_mode": _ROPE_MODE,
    }
    name = next(key for key, mapping in mappings.items() if mapping is names)
    return adapter._mode_int(name, value)


_W_QR_QUANT_MODE = {"MXFP8": 1, "HIF8": 2}
_QKV_QUANT_MODE = {
    "NO_QUANT": 0,
    "KV_FP8_PER_TOKEN_GROUP": 1,
    "Q_KV_FP8": 2,
    "Q_KV_HIF8": 3,
}
_CACHE_MODE = {"PA_BSND": 0, "PA_NZ": 1}
_ROPE_MODE = {"NONE": 0, "INTERLEAVE_HALF": 1, "HALF": 2}


def _ensure_torch(value):
    """将 NumPy 输入及语义 HiF8 载体适配为 Torch，已有 Torch 张量直接传递。"""
    if value is None:
        return value
    if torch.is_tensor(value):
        # 原生 HiF8 以 uint8/int8 单字节载体暴露给公开接口，不进行数值转换。

        return value.view(torch.uint8) if "hifloat" in str(value.dtype) else value
    arr = np.asarray(value)
    name = str(arr.dtype)
    if "hifloat8" in name:
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


def _golden_cache_snapshot(cache):
    """创建独立、连续的 CPU Cache 影子副本，保留 dtype 和各页字节。"""
    return cache.detach().cpu().contiguous().clone()


def descale_logical(descale, weight_format, output_size=None):
    """将声明为 NZ 的 MXFP8 权重 descale 映射为 CPU 参考的 ND 顺序。

    设备逻辑形状 (K//64,O,2) 转为 (O,K//64,2)，只改变索引顺序，不改变载荷字节。
    """
    if descale is None or not torch.is_tensor(descale) or descale.dim() != 3:
        return descale
    if str(weight_format).strip().upper() not in ("NZ", "FRACTAL_NZ", "29"):
        return descale
    if output_size is not None and descale.shape[0] == output_size:
        return descale.contiguous()
    return descale.permute(1, 0, 2).contiguous()


def quant_mla_prolog_golden(
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
    quant_scale_kv=None,
    rope_sin=None,
    rope_cos=None,
    *,
    norm_eps=1e-5,
    w_qr_quant_mode=1,
    qkv_quant_mode=0,
    cache_mode=0,
    qr_scale=1.0,
    kvr_scale=1.0,
    rope_mode=1,
    return_qr=False,
    weight_format=None,
    **kwargs,
):
    """使用与设备调用相同的参数规划执行 CPU Golden。"""
    # 遵循公开属性契约，拒绝未知属性、布尔模式和非法标量，避免静默使用默认值。

    import quant_mla_ttk_ops as _adapter

    _adapter._validate_unknown_attributes(kwargs)
    weight_format = _adapter._resolve_weight_format(
        (wqa, wqb, wkva), kwargs.get("tensor_formats"), weight_format
    )
    _adapter._is_nz(weight_format)  # 非法值直接报错，不静默按 ND 处理。
    _adapter._validate_scalars(norm_eps, qr_scale, kvr_scale)
    _adapter._validate_modes(
        qkv_quant_mode, w_qr_quant_mode, cache_mode, rope_mode, return_qr
    )
    reference = _load_reference()
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
    # 还原可选的 BF16 类型提升，使参考计算使用声明的输入类型。

    wkva, wkb = (_as_bf16(t) for t in (wkva, wkb))
    rope_sin, rope_cos = (_as_bf16(t) for t in (rope_sin, rope_cos))
    w_qr_quant_mode = _mode_int(w_qr_quant_mode, _W_QR_QUANT_MODE)
    # CSV 的 NZ descale 逻辑形状为 (K//64,O,2)，CPU 参考使用 (O,K//64,2)；只转置索引顺序，不改变数值或 dtype。

    descale_wqa, descale_wqb, descale_wkva = (
        descale_logical(t, weight_format, weight.shape[0])
        for t, weight in zip((descale_wqa, descale_wqb, descale_wkva), (wqa, wqb, wkva))
    )
    # Golden 使用独立的 CPU Cache，按既有 PA_BSND/PA_NZ 字节顺序整理各页，不修改 TTK 输入及底层存储。

    kv_cache = _golden_cache_snapshot(kv_cache)
    q, descale_q, qr, descale_qr = reference.quant_mla_prolog_golden(
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
        norm_eps=float(norm_eps),
        w_qr_quant_mode=w_qr_quant_mode,
        qkv_quant_mode=_mode_int(qkv_quant_mode, _QKV_QUANT_MODE),
        cache_mode=_mode_int(cache_mode, _CACHE_MODE),
        qr_scale=float(qr_scale),
        kvr_scale=float(kvr_scale),
        rope_mode=_mode_int(rope_mode, _ROPE_MODE),
        return_qr=bool(return_qr),
    )
    # descale_qr 保留原生 E8M0 语义 dtype，不降为 uint8；HiF8/E8M0 CSV 使用 en-dtypes 完成 NumPy 传输。

    native_e8 = getattr(torch, "float8_e8m0fnu", None)
    if w_qr_quant_mode == 1 and native_e8 is not None and descale_qr.dtype != native_e8:
        descale_qr = descale_qr.view(native_e8)
    # 参考按算子的物理布局原地更新 Cache，与包装器的第五个输出槽位对应。

    return [q, descale_q, qr, descale_qr, kv_cache]
