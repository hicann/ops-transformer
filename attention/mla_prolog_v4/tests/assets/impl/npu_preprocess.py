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

"""在计时和捕获前适配输入格式、记录 Cache 布局并执行非计算型契约检查。"""

import importlib


def _adapter():
    return importlib.import_module("mla_prolog_v4_ttk_ops")


def run(
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
    cache_mode="PA_BSND",
    weight_format=None,
    **kwargs,
):
    """在计时前准备输入格式并执行适用的契约检查。"""
    adapter = _adapter()
    weight_format = adapter._resolve_weight_format(
        (wqa, wqb, wkva), kwargs.get("tensor_formats"), weight_format
    )
    # 原生 HiF8 是传输类型，公开接口使用 uint8 字节载体。
    declared = kwargs.get("tensor_dtypes") or ()
    cache_dtype = declared[7] if len(declared) > 7 else None
    kv_cache = adapter._transport().prepare_device_cache(
        kv_cache, kwargs.get("testcase_name"), cache_dtype
    )
    # 在首次解析即时或图模式算子前加载模块，完成 torch.library schema 注册。

    adapter._load_under_test()
    # NZ 权重按逻辑 weight.T 转换为设备格式。
    import torch_npu

    if adapter._validate_weight_format(weight_format):
        for weight in (wqa, wqb, wkva):
            if not getattr(weight, "_ttk_nz_transposed", False):
                logical = torch_npu.npu_format_cast(weight, 2)
                weight.data = torch_npu.npu_format_cast(logical.t().contiguous(), 29)
                weight._ttk_nz_transposed = True
    qkv_quant_mode = kwargs.get("qkv_quant_mode", 0)
    qr_scale = kwargs.get("qr_scale", 1.0)
    kvr_scale = kwargs.get("kvr_scale", 1.0)
    rope_mode = kwargs.get("rope_mode", 1)
    return_qr = kwargs.get("return_qr", False)
    import logging

    logging.info("cache shape=%s stride=%s", tuple(kv_cache.shape), kv_cache.stride())
    from pathlib import Path
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        __name__ + "_audit", Path(__file__).with_name("audit.py")
    )
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    attrs = dict(
        norm_eps=norm_eps,
        qkv_quant_mode=qkv_quant_mode,
        cache_mode=cache_mode,
        qr_scale=qr_scale,
        kvr_scale=kvr_scale,
        rope_mode=rope_mode,
        return_qr=return_qr,
    )
    entry = adapter._load_under_test().mla_prolog_v2
    for key, enum in (
        ("qkv_quant_mode", "QkvQuantMode"),
        ("cache_mode", "CacheMode"),
        ("rope_mode", "RopeMode"),
    ):
        attrs[key] = getattr(entry, enum)(adapter._mode_int(key, attrs[key]))
    audit.run(
        adapter,
        [
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
        ],
        attrs,
    )
    return None
