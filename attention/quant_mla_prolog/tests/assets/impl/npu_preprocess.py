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
import logging


def _adapter():
    return importlib.import_module("quant_mla_ttk_ops")


def run(
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
    """校验并记录张量实际格式，保留绑定的张量对象。"""
    adapter = _adapter()
    weight_format = adapter._resolve_weight_format(
        (wqa, wqb, wkva), kwargs.get("tensor_formats"), weight_format
    )
    import torch
    import torch_npu

    # 即使 CSV 声明语义 HiF8，接口仍要求 uint8 载体。
    for value in (x, wqa, wqb, wkva):
        if adapter._transport().is_hif8(value):
            fmt = torch_npu.get_npu_format(value)
            raw = torch_npu.npu_format_cast(value, 2).view(torch.uint8)
            value.data = torch_npu.npu_format_cast(raw, fmt) if fmt == 29 else raw
    declared = kwargs.get("tensor_dtypes") or ()
    transport = adapter._transport()
    key = kwargs.get("testcase_name")
    if any("hifloat8" in str(d) for d in declared) and key not in transport._HOST_CACHE:
        transport.loaded_cache_host(kv_cache, key)
    kv_cache = adapter._transport().prepare_device_cache(
        kv_cache,
        kwargs.get("testcase_name"),
        declared[11] if len(declared) > 11 else None,
    )
    # 在首次解析即时或图模式算子前加载模块，完成 torch.library schema 注册。

    adapter._load_under_test()
    mode = adapter._mode_int("w_qr_quant_mode", w_qr_quant_mode)
    # 在计时和捕获前将 descale 从 (O,G,2) 适配为 (G,O,2)，保留 CSV 输入及绑定对象。

    if mode == 1 and adapter._is_nz(weight_format):
        import torch
        import torch_npu

        for weight, scale in zip(
            (wqa, wqb, wkva), (descale_wqa, descale_wqb, descale_wkva)
        ):
            if (
                not getattr(scale, "_ttk_descale_nz_prepared", False)
                and not getattr(weight, "_ttk_nz_transposed", False)
                and scale.dim() == 3
                and scale.shape[0] == weight.shape[0]
            ):
                logical = torch_npu.npu_format_cast(scale, 2)
                logical = logical.permute(1, 0, 2).contiguous()
                scale.data = torch_npu.npu_format_cast(
                    logical, 29, input_dtype=torch.float8_e8m0fnu
                )
                scale._ttk_descale_nz_prepared = True
    adapter._validate_device_formats(
        (wqa, wqb, wkva),
        (descale_wqa, descale_wqb, descale_wkva),
        descale_x,
        mode,
        weight_format,
    )
    # 字节载体打包需要 E8M0 input_dtype 提示，普通 uint8 的 format-29 打包会填充末轴；仅在计时前规范存储。

    if mode == 1 and adapter._is_nz(weight_format):
        import torch
        import torch_npu

        for scale in (descale_wqa, descale_wqb, descale_wkva):
            if torch_npu.get_storage_size(scale) != scale.numel():
                logical = torch_npu.npu_format_cast(scale, 2)
                packed = torch_npu.npu_format_cast(
                    logical, 29, input_dtype=torch.float8_e8m0fnu
                )
                if packed.untyped_storage().nbytes() != scale.numel():
                    raise ValueError("MX scale packing differs from pytest contract")
                scale.data = packed
    _log_inputs(
        adapter,
        kwargs.get("testcase_name", "?"),
        mode,
        weight_format,
        (
            ("x", x),
            ("wqa", wqa),
            ("wqb", wqb),
            ("wkva", wkva),
            ("descale_x", descale_x),
            ("descale_wqa", descale_wqa),
            ("descale_wqb", descale_wqb),
            ("descale_wkva", descale_wkva),
        ),
    )
    # MX scale 存储已在计时及捕获前完成规范化。
    # NZ 权重按逻辑 weight.T 转换为设备格式。
    if adapter._is_nz(weight_format):
        import torch_npu

        for weight in (wqa, wqb, wkva):
            if not getattr(weight, "_ttk_nz_transposed", False):
                logical = torch_npu.npu_format_cast(weight, 2)
                weight.data = torch_npu.npu_format_cast(logical.t().contiguous(), 29)
                weight._ttk_nz_transposed = True
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
        w_qr_quant_mode=w_qr_quant_mode,
    )
    sample = adapter._load_under_test()
    for key in ("w_qr_quant_mode", "qkv_quant_mode", "cache_mode", "rope_mode"):
        attrs[key] = adapter._mode_value(sample, key, attrs[key])
    attrs["_weight_format"] = weight_format
    audit.run(
        adapter,
        [
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
        ],
        attrs,
    )
    return None


def _log_inputs(adapter, testcase, mode, weight_format, tensors):
    """记录本次预处理传入算子的具体张量。

    dtype 和逻辑 shape 来自张量本身，format 来自 torch_npu.get_npu_format，不使用 msprof 字段。
    """
    declared = "NZ" if adapter._is_nz(weight_format) else "ND"
    logging.info(
        "quant_mla_prolog preprocess: testcase=%s w_qr_quant_mode=%d "
        "weight_format(CSV)=%s (actual formats below, not msprof fields)",
        testcase,
        mode,
        declared,
    )
    for name, tensor in tensors:
        if tensor is None or not hasattr(tensor, "shape"):
            continue
        fmt = adapter._device_format(tensor)
        logging.info(
            "quant_mla_prolog preprocess:   %-12s dtype=%-18s logical=%-20s format=%s",
            name,
            str(getattr(tensor, "dtype", "?")),
            tuple(tensor.shape),
            fmt if fmt is not None else "?",
        )
