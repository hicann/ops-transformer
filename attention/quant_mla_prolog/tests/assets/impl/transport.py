# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""适配 HiF8 实数 CSV、CPU 编码和 uint8 接口载体。

数值定义保留在 pytest_reference 中，此模块只连接 TTK 的传输表示。
"""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

_HOST_CACHE = {}


def reference():
    name = "quant_mla_prolog_pytest_9eeace4"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(
            name, Path(__file__).with_name("pytest_reference.py")
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def is_hif8(value):
    return value is not None and "hifloat8" in str(value.dtype)


def to_torch(value):
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.view(torch.uint8) if is_hif8(value) else value
    arr = np.asarray(value)
    name = str(arr.dtype)
    if "hifloat8" in name:
        return torch.from_numpy(arr.view(np.uint8))
    if "bfloat16" in name:
        return torch.from_numpy(arr.view(np.uint16)).view(torch.bfloat16)
    if name == "float8_e4m3fn":
        return torch.from_numpy(arr.view(np.uint8)).view(torch.float8_e4m3fn)
    return torch.from_numpy(arr)


def normalize_hif8(value):
    """按饱和、半值远离零舍入、NaN 转零及规范零编码处理 HiF8。

    有限编码的语义值重新编码具有幂等性；原始 uint8 不按实数输入解释。
    """
    if not is_hif8(value):
        return
    if torch.is_tensor(value):
        real = value.detach().cpu().float().contiguous()
    else:
        real = torch.from_numpy(
            np.ascontiguousarray(np.asarray(value).astype(np.float32))
        )
    codes = reference()._hif8_encode_raw(real).reshape(real.shape)
    if torch.is_tensor(value):
        value.view(torch.uint8).copy_(codes.to(value.device))
    else:
        np.asarray(value).view(np.uint8)[...] = codes.numpy()


def remember_cache(value, key):
    """为非原生 dtype 回退路径保留 CPU CSV 的底层存储和视图。

    框架可能物化为连续的原生 HiF8 张量；布局仅来自 CSV 生成的 ndarray，不设置固定间隔。
    """
    if torch.is_tensor(value):
        host = to_torch(value)
    else:
        from ttk.core_modules.common.backend.torch_backend import np_to_torch_inputs

        arr = np.asarray(value)
        root = arr
        while isinstance(root.base, np.ndarray):
            root = root.base
        raw = root.view(np.uint8) if is_hif8(arr) else root
        offset = (arr.ctypes.data - root.ctypes.data) // arr.dtype.itemsize
        strides = tuple(s // arr.dtype.itemsize for s in arr.strides)
        meta = SimpleNamespace(
            np_storages=[raw],
            flat_tensor_view_shapes=[arr.shape],
            flat_tensor_dtypes=["uint8" if is_hif8(arr) else str(arr.dtype)],
            flat_view_stride=lambda _: strides,
            flat_view_offset=lambda _: offset,
        )
        host = np_to_torch_inputs(meta, [arr.view(np.uint8) if is_hif8(arr) else arr])[
            0
        ]
    _HOST_CACHE[key] = host


def loaded_cache_host(value, key):
    """手动加载数据跳过输入钩子时，恢复 CSV 声明的布局。

    名称仅作为元数据查找键，不用于分类用例。
    """
    import ast
    import csv
    from ttk.core_modules.common.backend.torch_backend import np_to_torch_inputs

    matches = []
    for path in Path(__file__).parents[1].glob("*.csv"):
        for row in csv.DictReader(path.open(encoding="utf-8-sig")):
            if row.get("testcase_name") == key:
                matches.append(row)
    if len(matches) != 1:
        raise ValueError("Cannot resolve unique CSV layout for loaded HiF8 cache")
    row = matches[0]
    shape = ast.literal_eval(row["tensor_view_shapes"])[11]
    stores = (
        ast.literal_eval(row["tensor_storage_shapes"])
        if row.get("tensor_storage_shapes")
        else None
    )
    strides = (
        ast.literal_eval(row["tensor_view_strides"])
        if row.get("tensor_view_strides")
        else None
    )
    offsets = (
        ast.literal_eval(row["tensor_view_offsets"])
        if row.get("tensor_view_offsets")
        else None
    )
    storage_shape = stores[11] if stores and stores[11] is not None else shape
    stride = (
        strides[11]
        if strides and strides[11] is not None
        else tuple(np.empty(shape, dtype=np.uint8).strides)
    )
    offset = (offsets[11] or 0) if offsets else 0
    # 保护区按参考字节哨兵初始化；逻辑行从已加载输入复制，不重新采样或编码。

    import ml_dtypes

    dtype = ml_dtypes.bfloat16 if value.dtype == torch.bfloat16 else np.uint8
    storage = np.zeros(storage_shape, dtype=dtype)
    storage.view(np.uint8).fill(0x5A)
    meta = SimpleNamespace(
        np_storages=[storage],
        flat_tensor_view_shapes=[shape],
        flat_tensor_dtypes=["bfloat16" if value.dtype == torch.bfloat16 else "uint8"],
        flat_view_stride=lambda _: stride,
        flat_view_offset=lambda _: offset,
    )
    host = np_to_torch_inputs(meta, [storage])[0]
    host.copy_(to_torch(value).detach().cpu())
    _HOST_CACHE[key] = host
    return host


def prepare_device_cache(value, key, declared_dtype=None):
    if (
        not is_hif8(value)
        and "hifloat8" not in str(declared_dtype)
        and key not in _HOST_CACHE
    ):
        return to_torch(value)
    host = _HOST_CACHE.get(key)
    if host is None:
        host = loaded_cache_host(value, key)
    from ttk.core_modules.common.backend.npu_torch_backend import NpuTorchBackend

    prepared = NpuTorchBackend()._clone_full_storage(host, value.device)
    # 保留 TTK 绑定的同一参数对象，仅调整接口 dtype 与底层存储表示。
    value.data = prepared
    return value
