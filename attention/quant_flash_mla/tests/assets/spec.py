#!/usr/bin/python3
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


"""TTK TestSpec for quant_flash_mla.

Maps two ops:
  - torch.ops.cann_ops_transformer.quant_flash_mla
  - torch.ops.cann_ops_transformer.quant_flash_mla_metadata
"""

import importlib.util
import sys
from pathlib import Path

ASSET_IMPL_DIR = Path(__file__).with_name("impl")

_impl_cache = {}


def load_impl_module(stem):
    """Lazily load impl/{stem}.py under a stable synthetic name.

    Deferred import avoids importing torch_npu in the main process
    before TTK forks workers (causes NPU re-init errors).
    """
    mod_name = f"qfmla_assets_impl_{stem}"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = ASSET_IMPL_DIR / f"{stem}.py"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    _impl_cache[stem] = module
    return _impl_cache[stem]


def _xpu_reference(**kwargs):
    """XPU 三方参考: 固定 256 列分块的分段量化 golden (原生精度, 不升档).

    由远端 xpu_server 调用, 产出三方比对用的 third_party 数据.
    必须与 NPU 实现处在同一误差量级: 全局精确参考与本算子有确定的量化误差差,
    会让 cross_check 的 |b-g| 分母趋零, 比值恒定爆掉; 分段参考复现 P 量化噪声,
    分母才是有效的噪声地板.
    """
    return load_impl_module("golden").cpu_qfmla(
        _force_global_reference=True, _sectioned_reference=True, **kwargs
    )


def _resolve_quant_mode(args, kwargs):
    """quant_mode is positional arg[6] after op-schema direct call."""
    qm = args[6] if len(args) > 6 else kwargs.get("quant_mode", 1)
    return int(qm) if qm is not None else 1


def _golden_dispatch(*args, **kwargs):
    return load_impl_module("golden").cpu_qfmla(*args, **kwargs)


def _inputs_dispatch(*args, **kwargs):
    return load_impl_module("inputs").generate_qfmla_inputs(*args, **kwargs)


def _compare_dispatch(*outputs, **kwargs):
    return load_impl_module("compare").compare(*outputs, **kwargs)


def _cross_check_active():
    """CLI 是否启用三方比对 (--compare cross_check)。"""
    try:
        from ttk.switches import get_global_storage
    except Exception:  # pylint: disable=broad-except
        return False
    try:
        method = getattr(get_global_storage(), "compare_method", None)
    except Exception:  # pylint: disable=broad-except
        return False
    return method is not None and str(method).lower() == "cross_check"


class _CompareHook:
    """compare 钩子的动态解析。

    TTK 的自定义 compare 只收到 outputs/goldens, 拿不到 third_parties
    (comparison.py: invoke_custom_compare), 且返回 None 会报
    "compare must return dict or list[dict]"。
    因此三方比对时把 compare 解析成 None, 让框架退回标准 compare,
    由 CrossCheckComparison 用 golden(升档真值) + third_party(XPU) 判定;
    两方比对仍走 impl/compare.py。
    """

    def __init__(self, func):
        self._func = func

    def __get__(self, instance, owner):
        return None if _cross_check_active() else self._func


class QuantMlaSpec:
    """quant_flash_mla test spec (FP8_E4M3 full-quant).

    metadata 由 npu_preprocess 在主算子调用前生成并回填 metadata slot,
    主算子直接由 ttk 调用 torch.ops.cann_ops_transformer.quant_flash_mla。

    golden/customize_inputs/compare/npu_preprocess 均懒加载 impl 模块,
    避免主进程 fork 前 import torch_npu (与 flash_attn 资产一致)。
    """

    golden = staticmethod(_golden_dispatch)
    customize_inputs = staticmethod(_inputs_dispatch)

    # 两方→impl/compare.py; 三方(--compare cross_check)→解析成 None, 走框架标准 compare。
    compare = _CompareHook(_compare_dispatch)

    @staticmethod
    def npu_preprocess(*args, **kwargs):
        return load_impl_module("npu_preprocess").run(*args, **kwargs)

    torch_graph = load_impl_module("graph").QuantMlaAclGraph

    # XPU 三方参考: 复用 CPU 参考实现, 输出原生精度(不升档)。
    # 由远端 xpu_server 调用; 函数体内延迟 import, 避免主进程提前拉起 torch_npu。
    third_party = {"torch": _xpu_reference}

    # 两方走 impl/compare.py; 三方(--compare cross_check)由框架原生比对,
    # 判据取 level=L2 (mare/mere/rmse = 2.0/1.2/1.2)。
    tolerance = {
        "float16": {
            "rtol": 0.005,
            "ptol": 0.005,
            "atol": 0.000025,
            "level": "L2",
        },
        "bfloat16": {
            "rtol": 0.0078125,
            "ptol": 0.005,
            "atol": 0.0001,
            "level": "L2",
        },
        "float32": {
            "rtol": 0.005,
            "ptol": 0.005,
            "atol": 0.000025,
            "level": "L2",
        },
    }


class QuantMlaMetadataSpec:
    """quant_flash_mla_metadata spec (metadata-only op)."""

    @staticmethod
    def customize_inputs(*args, **kwargs):
        return load_impl_module(
            "metadata_inputs"
        ).generate_quant_flash_mla_metadata_inputs(*args, **kwargs)


__spec__ = {
    "torch.ops.cann_ops_transformer.quant_flash_mla": "QuantMlaSpec",
    "torch.ops.cann_ops_transformer.quant_flash_mla_metadata": "QuantMlaMetadataSpec",
}
