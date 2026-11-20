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

"""准备 Quant MLA 用例的 CPU 输入。

输入张量与生成的 NumPy 存储共享内存，在 Golden 前原地改写，返回值不被使用。
cache_index_rule 指定槽位：unique 为每个 token 分配唯一槽位；negative 全部为 -1；
duplicate 全部为 0，用于多 token 的重复槽位检查；out_of_range 为容量上界，用于越界检查。
合法非负槽位必须唯一且小于 block_number*block_size。
rope_sin/rope_cos 保持原值，设备和参考以相同算术使用这些张量。
"""

import importlib.util
import os
import sys
import zlib

import numpy as np
import torch


def shift(value, salt):
    return ((value * 0x9E3779B1) ^ (salt * 0x85EBCA77)) & 0xFFFFFFFF


_IMPL_DIR = os.path.dirname(os.path.abspath(__file__))
_REFERENCE_MODULE = "quant_mla_prolog_golden_reference"
_HIF8_X_SIGMA = 0.3  # 激活使用 randn*0.3。
_HIF8_W_SIGMA = 0.1  # 权重使用 randn*0.1。
_HIF8_CACHE_FILL = 0x35  # uint8 Cache 使用 0x35 字节初始化。


def _golden_reference():
    """每个进程仅加载一次本地 Golden，与 Golden 钩子共享模块。"""
    module = sys.modules.get(_REFERENCE_MODULE)
    if module is not None:
        return module
    path = os.path.join(_IMPL_DIR, "golden_reference.py")
    spec = importlib.util.spec_from_file_location(_REFERENCE_MODULE, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[_REFERENCE_MODULE] = module
    spec.loader.exec_module(module)
    return module


def _stable_seed(tensor, salt):
    """由固定基数 42、张量形状和 salt 生成幂等随机种子。

    种子不依赖会被覆盖的张量内容，保证同进程重复调用及即时/图执行得到相同输入。
    """
    base = 42
    shape_hash = zlib.crc32(("%s|%s" % (tuple(tensor.shape), tensor.dtype)).encode())
    return (base ^ shift(shape_hash, salt)) & 0xFFFFFFFF


def _pytest_hif8(tensor, sigma, salt):
    """按 randn*sigma、饱和及半值远离零舍入生成 HiF8 字节。

    编码在 ±32768 饱和，NaN 和舍入至零的有限值均映射为 0x00。
    载体不会包含 0x80 的 NaN 或 0x6F/0xEF 的正负 Inf。
    """
    g = _golden_reference()
    generator = torch.Generator().manual_seed(_stable_seed(tensor, salt))
    values = torch.randn(tuple(tensor.shape), dtype=torch.float32, generator=generator)
    codes = g._hif8_encode_raw(values * sigma).reshape(tuple(tensor.shape))
    if torch.is_tensor(tensor):
        tensor.copy_(codes.view(torch.uint8).reshape(tuple(tensor.shape)))
    else:
        tensor.view(np.uint8)[...] = codes.numpy().view(np.uint8)


def _pytest_hif8_descales(descale_x, descale_wqa, descale_wqb, descale_wkva):
    """激活 descale 为 [0.5]，权重 descale 为 linspace(0.5,1.5,O) 的 (1,O) 视图。"""
    descale_x.fill_(0.5)
    for scale in (descale_wqa, descale_wqb, descale_wkva):
        width = int(scale.shape[-1])
        scale.copy_(
            torch.linspace(0.5, 1.5, width, dtype=torch.float32)
            .reshape(1, width)
            .expand_as(scale)
        )


def _assign(target, values):
    """原地写入 Torch 张量或 NumPy 回退数组。"""
    if torch.is_tensor(target):
        target.copy_(torch.as_tensor(values, dtype=target.dtype))
    else:
        target[...] = np.asarray(values, dtype=target.dtype)


# 按激活及三个权重的抽样顺序设置参考编码范围。
# 激活编码 randint(124,128)，即 124..127，不含 NaN。
# 权重编码 randint(125,129)，即 125..128，包含 NaN 编码 0x80。
_E8M0_CODE_RANGES_PYTEST = ((124, 127), (125, 128), (125, 128), (125, 128))
_E8M0_CODE_RANGES_UNIFORM = ((120, 128),) * 4
_E8M0_CODE_RANGES_FIXED = ((120, 128),) * 4  # 四个 descale 的统一编码范围为 120..128。


def _is_e8m0_like(tensor):
    return tensor is not None and "e8m0" in str(getattr(tensor, "dtype", ""))


def _e8m0_ranges_from(input_ranges, fallback):
    """按槽位读取 E8M0 编码范围，优先使用 CSV 声明。

    input_data_ranges 的 5..8 槽位对应四个 descale；此钩子覆盖 TTK 生成的 scale，写入声明的编码。
    """
    ranges = []
    for offset, index in enumerate((5, 6, 7, 8)):
        declared = None
        if input_ranges and index < len(input_ranges):
            declared = input_ranges[index]
        if isinstance(declared, (list, tuple)) and len(declared) >= 2:
            low, high = int(declared[0]), int(declared[1])
            ranges.append((low, high))
        else:
            ranges.append(fallback[offset])
    return tuple(ranges)


def _rewrite_e8m0_codes(descales, ranges):
    """向原生 E8M0 descale 张量写入原始字节编码。

    TTK 将 input_data_ranges 解释为 scale 数值，编码范围 (120,128) 会被量化到 2^6 附近。
    此钩子按形状生成确定性、幂等的字节编码，覆盖已生成的 scale，得到 0x78..0x80 载荷。
    """
    for salt, (tensor, (low, high)) in enumerate(zip(descales, ranges)):
        if not _is_e8m0_like(tensor):
            continue
        shape = tuple(tensor.shape)
        rng = np.random.default_rng(_stable_seed(tensor, 0xE8 + salt))
        codes = rng.integers(low, high + 1, size=shape, dtype=np.uint8)
        if torch.is_tensor(tensor):
            tensor.view(torch.uint8)[...] = torch.as_tensor(codes)
        else:
            tensor.view(np.uint8)[...] = codes


def _rewrite_cache_index(cache_index, kv_cache, rule):
    tokens = int(np.prod(cache_index.shape))
    block_number, block_size = int(kv_cache.shape[0]), int(kv_cache.shape[1])
    capacity = block_number * block_size
    if rule in (None, "", "unique"):
        if tokens > capacity:
            return
        _assign(cache_index, np.arange(tokens, dtype=np.int64))
    elif rule == "negative":
        _assign(cache_index, np.full(cache_index.shape, -1, dtype=np.int64))
    elif rule == "duplicate":
        _assign(cache_index, np.zeros(cache_index.shape, dtype=np.int64))
    elif rule == "out_of_range":
        _assign(cache_index, np.full(cache_index.shape, capacity, dtype=np.int64))
    else:
        raise ValueError(f"unsupported cache_index_rule: {rule}")


_E8M0_STYLES = ("fixed", "csv", "pytest", "ttk", "code", "scale")
_HIF8_STYLES = ("range", "uniform", "pytest")


def _validate_styles(e8m0_input_style, hif8_input_style):
    """校验造数风格属性，非法值直接报错，不静默退回默认值。"""
    e8m0_style = str(e8m0_input_style).strip().lower()
    if e8m0_style not in _E8M0_STYLES:
        raise ValueError(
            "unsupported e8m0_input_style: %r (expected fixed/csv/pytest/ttk/code/scale)"
            % (e8m0_input_style,)
        )
    hif8_style = str(hif8_input_style).strip().lower()
    if hif8_style not in _HIF8_STYLES:
        raise ValueError(
            "unsupported hif8_input_style: %r (expected range/uniform/pytest)"
            % (hif8_input_style,)
        )
    return e8m0_style, hif8_style


def customize_inputs(
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
    cache_index_rule="unique",
    hif8_input_style="range",
    e8m0_input_style="fixed",
    **kwargs,
):
    """按规则准备 cache_index、HiF8 输入及 E8M0 descale。

    hif8_input_style：range 为默认值，保留 CSV 实数范围生成的 HiF8；
    pytest 使用 SAT+RNA(randn*0.3/randn*0.1)、[0.5]/linspace(0.5,1.5,O) 和 0x35 Cache；
    uniform 是 range 的兼容名称。

    e8m0_input_style：fixed 默认对四个 descale 写入编码 120..128；csv 使用 5..8 槽位范围；
    ttk 保留框架结果，仅适用于框架按编码采样的情形；pytest 对激活使用 124..127，
    对权重使用 125..128，后者包含 NaN 编码 0x80；code 对四槽使用统一编码范围 120..128；
    scale 保留框架按实数 scale 生成的结果，例如范围 (0.0078125,2.0)。
    编码通过 uint8 视图写入，dtype 保持为原生 float8_e8m0。
    """
    import quant_mla_ttk_ops as adapter

    transport = adapter._transport()
    for value in (x, wqa, wqb, wkva, kv_cache):
        if transport.is_hif8(value):
            transport.normalize_hif8(value)
    if any(transport.is_hif8(v) for v in (x, wqa, wqb, wkva, kv_cache)):
        transport.remember_cache(kv_cache, kwargs.get("testcase_name"))

    def _is_tensor(v):
        return torch.is_tensor(v) or isinstance(v, np.ndarray)

    if _is_tensor(cache_index) and _is_tensor(kv_cache):
        _rewrite_cache_index(cache_index, kv_cache, cache_index_rule)

    e8m0_style, hif8_style = _validate_styles(e8m0_input_style, hif8_input_style)
    if e8m0_style in ("fixed", "csv", "pytest", "code") and w_qr_quant_mode == 1:
        if e8m0_style == "fixed":
            ranges = _E8M0_CODE_RANGES_FIXED
        elif e8m0_style == "csv":
            ranges = _e8m0_ranges_from(
                kwargs.get("input_ranges"), _E8M0_CODE_RANGES_PYTEST
            )
        elif e8m0_style == "pytest":
            ranges = _E8M0_CODE_RANGES_PYTEST
        else:
            ranges = _E8M0_CODE_RANGES_UNIFORM
        _rewrite_e8m0_codes((descale_x, descale_wqa, descale_wqb, descale_wkva), ranges)
    if hif8_style == "pytest" and w_qr_quant_mode == 2:
        for salt, (tensor, sigma) in enumerate(
            (
                (x, _HIF8_X_SIGMA),
                (wqa, _HIF8_W_SIGMA),
                (wqb, _HIF8_W_SIGMA),
                (wkva, _HIF8_W_SIGMA),
            )
        ):
            if _is_tensor(tensor):
                _pytest_hif8(tensor, sigma, salt + 1)
        if all(
            torch.is_tensor(t)
            for t in (descale_x, descale_wqa, descale_wqb, descale_wkva)
        ):
            _pytest_hif8_descales(descale_x, descale_wqa, descale_wqb, descale_wkva)
        if qkv_quant_mode == 3 and torch.is_tensor(kv_cache):
            kv_cache.fill_(_HIF8_CACHE_FILL)
